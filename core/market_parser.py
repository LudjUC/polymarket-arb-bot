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
        rejected = 0
        for raw in raw_markets:
            mc = self._parse_market(raw)
            if mc:
                self._markets[mc.market_id] = mc
                parsed.append(mc)
            else:
                rejected += 1

        self._last_refresh = time.time()
        METRICS.gauge("active_markets", len(parsed))

        if parsed:
            log.info(
                "Loaded %d tradeable BTC markets (%d rejected). "
                "Run `python debug_markets.py` to see rejection reasons.",
                len(parsed), rejected,
            )
            for mc in parsed:
                log.info(
                    "  ✓ [%s]  yes=%.3f  liq=$%.0f  %s",
                    mc.condition_type, mc.yes_price, mc.liquidity_usd,
                    mc.question[:65],
                )
        else:
            log.warning(
                "0 tradeable BTC markets from %d returned. "
                "All %d were rejected. Run `python debug_markets.py` for details.",
                len(raw_markets), rejected,
            )
        return parsed

    def get_markets(self) -> List[MarketCondition]:
        return list(self._markets.values())

    def needs_refresh(self) -> bool:
        elapsed = time.time() - self._last_refresh
        return elapsed >= CONFIG.polymarket.market_refresh_interval_s

    # ------------------------------------------------------------------
    def _parse_market(self, raw: Dict) -> Optional[MarketCondition]:
        """
        Parse one raw Gamma API market dict into a MarketCondition.

        Handles every Polymarket schema variant observed in the wild:
          - tokens as a JSON string or a list
          - outcomePrices as a JSON string or a list
          - prices embedded in token dicts OR in outcomePrices
          - outcomes named "Yes"/"No", "YES"/"NO", or positional [0]=yes [1]=no
          - binary markets with exactly 2 outcomes
          - missing / zero expiry dates
          - liquidity as a string float or a number
        """
        import json as _json

        question = raw.get("question", "")
        if not any(w in question.lower() for w in ["btc", "bitcoin"]):
            return None

        threshold, ctype = _parse_threshold(question)
        if threshold is None:
            log.debug("No threshold parsed: %r", question[:80])

        # ── market ID ────────────────────────────────────────────────
        market_id = (
            str(raw.get("id", ""))
            or str(raw.get("conditionId", ""))
            or str(raw.get("condition_id", ""))
            or ""
        )

        # ── tokens ───────────────────────────────────────────────────
        tokens_raw = raw.get("tokens", [])
        if isinstance(tokens_raw, str):
            try:
                tokens_raw = _json.loads(tokens_raw)
            except Exception:
                tokens_raw = []
        tokens: List[Dict] = tokens_raw if isinstance(tokens_raw, list) else []

        # Strategy 1: match by outcome label "yes"/"no"
        yes_token = next(
            (t for t in tokens if str(t.get("outcome", "")).lower() == "yes"), None
        )
        no_token = next(
            (t for t in tokens if str(t.get("outcome", "")).lower() == "no"), None
        )

        # Strategy 2: positional — Polymarket binary markets always put YES first
        if (not yes_token or not no_token) and len(tokens) == 2:
            yes_token = tokens[0]
            no_token  = tokens[1]

        # Strategy 3: synthesise minimal token dicts from outcomes / outcomePrices
        outcomes_raw = raw.get("outcomes", "[]")
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = _json.loads(outcomes_raw)
            except Exception:
                outcomes_raw = []

        if (not yes_token or not no_token) and isinstance(outcomes_raw, list) and len(outcomes_raw) >= 2:
            yes_token = {"outcome": outcomes_raw[0], "token_id": "", "price": 0}
            no_token  = {"outcome": outcomes_raw[1], "token_id": "", "price": 0}

        if not yes_token or not no_token:
            log.debug("Skipping market %s — no YES/NO tokens: %r", market_id[:12], question[:60])
            return None

        # ── prices ───────────────────────────────────────────────────
        # Try outcomePrices array first (most reliable source in Gamma v2)
        outcome_prices_raw = raw.get("outcomePrices", "[]")
        if isinstance(outcome_prices_raw, str):
            try:
                op = [float(p) for p in _json.loads(outcome_prices_raw)]
            except Exception:
                op = []
        elif isinstance(outcome_prices_raw, list):
            try:
                op = [float(p) for p in outcome_prices_raw]
            except Exception:
                op = []
        else:
            op = []

        # outcomePrices[0] = YES, outcomePrices[1] = NO (Polymarket convention)
        yes_price = op[0] if len(op) >= 1 else 0.0
        no_price  = op[1] if len(op) >= 2 else 0.0

        # Fall back to per-token price field
        if yes_price == 0.0:
            yes_price = float(yes_token.get("price", 0) or 0)
        if no_price == 0.0:
            no_price = float(no_token.get("price", 0) or 0)

        # Last resort: if we have one price, infer the other
        if yes_price > 0 and no_price == 0.0:
            no_price = round(1.0 - yes_price, 4)
        elif no_price > 0 and yes_price == 0.0:
            yes_price = round(1.0 - no_price, 4)

        if yes_price + no_price > 0 and not (0.80 <= yes_price + no_price <= 1.20):
            log.debug(
                "Suspicious price sum for %s: yes=%.3f no=%.3f sum=%.3f",
                market_id[:12], yes_price, no_price, yes_price + no_price,
            )

        # ── liquidity ────────────────────────────────────────────────
        liq_raw = raw.get("liquidity", 0) or raw.get("volume", 0) or 0
        try:
            liquidity = float(liq_raw)
        except (TypeError, ValueError):
            liquidity = 0.0

        # ── expiry ───────────────────────────────────────────────────
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
            yes_token_id=str(yes_token.get("token_id", "") or ""),
            no_token_id=str(no_token.get("token_id", "") or ""),
            liquidity_usd=liquidity,
            raw=raw,
        )

        if not mc.tradeable:
            # Log the specific reason so we can tune config.py thresholds
            reasons = []
            if mc.expiry_ts != 0 and mc.is_expired:
                reasons.append("expired")
            elif mc.expiry_ts != 0 and mc.seconds_to_expiry < CONFIG.polymarket.min_time_to_expiry_s:
                reasons.append(f"expiry_too_soon({mc.seconds_to_expiry/60:.0f}m)")
            if not mc.has_sufficient_liquidity:
                reasons.append(f"low_liquidity(${liquidity:.0f})")
            if yes_price <= 0:
                reasons.append("yes_price_zero")
            if no_price <= 0:
                reasons.append("no_price_zero")
            log.debug(
                "Not tradeable [%s]: %s | %r",
                market_id[:12], ", ".join(reasons) or "unknown", question[:60],
            )
            return None

        return mc
