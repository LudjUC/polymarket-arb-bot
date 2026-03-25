"""
core/market_parser.py — Fetch, filter, and parse Polymarket BTC markets.

Polymarket has two APIs:
  - Gamma API: metadata, search, categories
  - CLOB API: real-time orderbook / prices

This module fetches markets, filters for BTC threshold conditions,
and returns structured MarketCondition objects ready for the engine.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from config import CONFIG
from utils.logger import get_logger
from utils.metrics import METRICS

log = get_logger("market_parser")


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass
class MarketToken:
    token_id: str
    outcome: str        # "Yes" or "No"
    price: float        # 0.0 – 1.0  (CLOB mid price)
    size_available: float  # Best ask size in USDC

@dataclass
class MarketCondition:
    market_id: str
    question: str
    condition_type: str     # "price_above" | "price_below" | "unknown"
    threshold_usd: Optional[float]
    expiry_ts: int          # Unix timestamp
    yes_price: float        # Polymarket YES token price (0–1)
    no_price: float
    yes_token_id: str
    no_token_id: str
    liquidity_usd: float
    raw: Dict               # Full API response for debugging

    @property
    def seconds_to_expiry(self) -> float:
        return self.expiry_ts - time.time()

    @property
    def is_expired(self) -> bool:
        return self.seconds_to_expiry <= 0

    @property
    def has_sufficient_liquidity(self) -> bool:
        return self.liquidity_usd >= CONFIG.polymarket.min_market_liquidity_usd

    @property
    def tradeable(self) -> bool:
        # expiry_ts == 0 means the API didn't return an end date; treat as far future
        expiry_ok = (
            self.expiry_ts == 0
            or (not self.is_expired and self.seconds_to_expiry >= CONFIG.polymarket.min_time_to_expiry_s)
        )
        return (
            expiry_ok
            and self.has_sufficient_liquidity
            and self.yes_price > 0
            and self.no_price > 0
        )


# ------------------------------------------------------------------
# Parser helpers
# ------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Threshold parsing — handles real Polymarket question formats such as:
#   "Will Bitcoin reach $105,000 by March 2025?"
#   "Will BTC be above $80k on January 1?"
#   "Bitcoin above $200,000 in 2025?"
#   "Will BTC fall below $50,000?"
#   "Bitcoin at $100000 end of year?"
# ---------------------------------------------------------------------------

# Pattern A: directional keywords followed by a dollar amount
_ABOVE_PATTERN = re.compile(
    r"(?:above|over|exceed|reach|hit|past|greater than|at least|>)\s*\$?([\d,]+)\s*(k|K)?",
    re.I,
)
_BELOW_PATTERN = re.compile(
    r"(?:below|under|drop(?:\s+to)?|fall(?:\s+to)?|less than|<)\s*\$?([\d,]+)\s*(k|K)?",
    re.I,
)
# Pattern B: bare dollar amount anywhere in question (fallback)
_BARE_DOLLAR_PATTERN = re.compile(r"\$\s*([\d,]+)\s*(k|K)?", re.I)


def _parse_threshold(question: str) -> tuple[Optional[float], str]:
    """
    Returns (threshold_usd, condition_type).
    condition_type: "price_above" | "price_below" | "unknown"

    Strategy:
      1. Try explicit below-keywords first (they're unambiguous).
      2. Try explicit above-keywords.
      3. Fall back to bare dollar amount and assume "price_above"
         (the most common Polymarket BTC question type).
    """
    def _to_usd(digits: str, k_suffix: Optional[str]) -> float:
        val = float(digits.replace(",", ""))
        if k_suffix:
            val *= 1_000
        # Heuristic: raw numbers < 1000 with no k suffix are probably
        # shorthand thousands (e.g. "100" meaning $100,000 in some questions).
        # We leave them as-is and let the condition engine filter implausible values.
        return val

    # 1. Check for below keywords
    m = _BELOW_PATTERN.search(question)
    if m:
        try:
            return _to_usd(m.group(1), m.group(2)), "price_below"
        except (ValueError, IndexError):
            pass

    # 2. Check for above keywords
    m = _ABOVE_PATTERN.search(question)
    if m:
        try:
            return _to_usd(m.group(1), m.group(2)), "price_above"
        except (ValueError, IndexError):
            pass

    # 3. Bare dollar sign fallback
    m = _BARE_DOLLAR_PATTERN.search(question)
    if m:
        try:
            val = _to_usd(m.group(1), m.group(2))
            # Only accept plausible BTC price ranges: $1k – $10M
            if 1_000 <= val <= 10_000_000:
                return val, "price_above"
        except (ValueError, IndexError):
            pass

    return None, "unknown"


def _parse_expiry(end_date_iso: Optional[str]) -> int:
    """Parse ISO 8601 date string to Unix timestamp."""
    if not end_date_iso:
        return 0
    try:
        dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return 0


# ------------------------------------------------------------------
# API client
# ------------------------------------------------------------------

class PolymarketClient:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-arb-bot/1.0"})
        self._cfg = CONFIG.polymarket

    def _get(self, base: str, path: str, params: Optional[Dict] = None, timeout: int = 10) -> Optional[Dict]:
        url = f"{base}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.error("API request failed: %s %s | %s", base, path, exc)
            METRICS.inc("api_errors")
            return None

    def search_btc_markets(self) -> List[Dict]:
        """
        Search Gamma API for active BTC price-related prediction markets.
        Fetches both 'bitcoin' and 'btc' keyword results and deduplicates.

        Gamma API quirks:
          - Boolean params must be sent as strings "true"/"false"
          - Response is a plain JSON array, not {"markets": [...]}
        """
        seen_ids: set = set()
        combined: List[Dict] = []

        for keyword in ("bitcoin", "btc"):
            data = self._get(
                self._cfg.gamma_api_url,
                "/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": "100",
                    "question": keyword,
                },
            )
            if not data:
                continue
            markets = data if isinstance(data, list) else data.get("markets", [])
            for m in markets:
                mid = m.get("id") or m.get("conditionId") or m.get("condition_id", "")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    combined.append(m)

        log.info("Gamma API returned %d unique BTC-related markets", len(combined))
        return combined

    def get_clob_price(self, token_id: str) -> Optional[float]:
        """Fetch mid-price for a token from the CLOB."""
        data = self._get(
            self._cfg.clob_api_url,
            f"/midpoint",
            params={"token_id": token_id},
        )
        if data and "mid" in data:
            try:
                return float(data["mid"])
            except (TypeError, ValueError):
                pass
        return None

    def get_order_book(self, token_id: str) -> Optional[Dict]:
        """Fetch order book for liquidity analysis."""
        return self._get(
            self._cfg.clob_api_url,
            f"/book",
            params={"token_id": token_id},
        )


# ------------------------------------------------------------------
# Market parser
# ------------------------------------------------------------------

class MarketParser:
    """
    Periodically fetches Polymarket markets, parses conditions,
    and maintains an up-to-date list of MarketCondition objects.
    """

    def __init__(self) -> None:
        self._client = PolymarketClient()
        self._markets: Dict[str, MarketCondition] = {}  # market_id → condition
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------
    def refresh(self) -> List[MarketCondition]:
        """Fetch and parse markets. Call periodically from main loop."""
        raw_markets = self._client.search_btc_markets()
        parsed = []
        for raw in raw_markets:
            mc = self._parse_market(raw)
            if mc:
                self._markets[mc.market_id] = mc
                parsed.append(mc)

        self._last_refresh = time.time()
        METRICS.gauge("active_markets", len(parsed))
        log.info("Parsed %d actionable BTC markets", len(parsed))
        return parsed

    def get_markets(self) -> List[MarketCondition]:
        return list(self._markets.values())

    def needs_refresh(self) -> bool:
        elapsed = time.time() - self._last_refresh
        return elapsed >= CONFIG.polymarket.market_refresh_interval_s

    # ------------------------------------------------------------------
    def _parse_market(self, raw: Dict) -> Optional[MarketCondition]:
        question = raw.get("question", "")
        # Quick filter: must mention BTC / Bitcoin
        if not any(w in question.lower() for w in ["btc", "bitcoin"]):
            return None

        threshold, ctype = _parse_threshold(question)
        if threshold is None:
            log.debug("No threshold parsed from: %r", question[:80])

        # --- Resolve market ID (Gamma uses several field names)
        market_id = (
            raw.get("id")
            or raw.get("conditionId")
            or raw.get("condition_id")
            or ""
        )

        # --- Resolve tokens
        # Gamma v2 embeds tokens as a JSON string in "tokens" field or a list
        tokens_raw = raw.get("tokens", [])
        if isinstance(tokens_raw, str):
            import json as _json
            try:
                tokens_raw = _json.loads(tokens_raw)
            except Exception:
                tokens_raw = []

        tokens: List[Dict] = tokens_raw if isinstance(tokens_raw, list) else []

        yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), None)
        no_token = next((t for t in tokens if t.get("outcome", "").lower() == "no"), None)

        if not yes_token or not no_token:
            return None

        # --- Resolve prices
        # Gamma embeds prices in token dicts OR in a parallel "outcomePrices" array
        outcome_prices_raw = raw.get("outcomePrices", "[]")
        if isinstance(outcome_prices_raw, str):
            import json as _json
            try:
                outcome_prices = [float(p) for p in _json.loads(outcome_prices_raw)]
            except Exception:
                outcome_prices = []
        elif isinstance(outcome_prices_raw, list):
            try:
                outcome_prices = [float(p) for p in outcome_prices_raw]
            except Exception:
                outcome_prices = []
        else:
            outcome_prices = []

        yes_price = float(yes_token.get("price", 0) or 0)
        no_price  = float(no_token.get("price", 0) or 0)

        # Fall back to outcomePrices array [yes_price, no_price]
        if yes_price == 0.0 and len(outcome_prices) >= 2:
            yes_price = outcome_prices[0]
            no_price  = outcome_prices[1]

        # Sanity check: prices should roughly sum to ~1 (allowing spread)
        if yes_price + no_price > 0 and not (0.8 <= yes_price + no_price <= 1.2):
            log.debug(
                "Suspicious prices for market %s: yes=%.3f no=%.3f",
                market_id, yes_price, no_price,
            )

        liquidity = float(raw.get("liquidity", 0) or 0)

        # --- Resolve expiry
        expiry_field = (
            raw.get("endDate")
            or raw.get("end_date_iso")
            or raw.get("endDateIso")
            or raw.get("gameStartTime")
        )

        mc = MarketCondition(
            market_id=market_id,
            question=question,
            condition_type=ctype,
            threshold_usd=threshold,
            expiry_ts=_parse_expiry(expiry_field),
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=yes_token.get("token_id", ""),
            no_token_id=no_token.get("token_id", ""),
            liquidity_usd=liquidity,
            raw=raw,
        )
        return mc if mc.tradeable else None
