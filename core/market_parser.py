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
        return (
            not self.is_expired
            and self.seconds_to_expiry >= CONFIG.polymarket.min_time_to_expiry_s
            and self.has_sufficient_liquidity
            and self.yes_price > 0
            and self.no_price > 0
        )


# ------------------------------------------------------------------
# Parser helpers
# ------------------------------------------------------------------

# Patterns: "Will BTC reach $105,000", "BTC above 80k", "Bitcoin > 90000"
_THRESHOLD_PATTERNS = [
    re.compile(r"(?:above|over|exceed|reach|hit|past|greater than|>)\s*\$?([\d,]+)k?", re.I),
    re.compile(r"\$?([\d,]+)k?\s*(?:or more|or higher|threshold)", re.I),
    re.compile(r"(?:below|under|drop.*?to|less than|<)\s*\$?([\d,]+)k?", re.I),
    re.compile(r"btc.*?\$\s*([\d,]+)", re.I),
]

def _parse_threshold(question: str) -> tuple[Optional[float], str]:
    """
    Returns (threshold_usd, condition_type).
    condition_type: "price_above" | "price_below" | "unknown"
    """
    q = question.lower()
    is_below = any(w in q for w in ["below", "under", "drop", "less than"])
    for pat in _THRESHOLD_PATTERNS:
        m = pat.search(question)
        if m:
            raw = m.group(1).replace(",", "")
            # Handle "80k" → 80_000
            multiplier = 1000 if q[m.end() - 1 : m.end()] == "k" or (m.end() < len(q) and q[m.end()] == "k") else 1
            try:
                val = float(raw) * multiplier
                ctype = "price_below" if is_below else "price_above"
                return val, ctype
            except ValueError:
                continue
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
        """Search Gamma API for active BTC price-related prediction markets."""
        data = self._get(
            self._cfg.gamma_api_url,
            "/markets",
            params={
                "tag": "crypto",
                "active": True,
                "closed": False,
                "limit": 100,
                # keyword search
                "question": "bitcoin",
            },
        )
        if not data:
            return []
        markets = data if isinstance(data, list) else data.get("markets", [])
        log.info("Gamma API returned %d BTC-related markets", len(markets))
        return markets

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
        # Quick filter: must mention BTC / Bitcoin and a dollar amount
        if not any(w in question.lower() for w in ["btc", "bitcoin"]):
            return None

        threshold, ctype = _parse_threshold(question)
        if threshold is None and ctype == "unknown":
            # Still include markets even if we can't parse threshold —
            # the condition engine will skip them gracefully
            pass

        tokens: List[Dict] = raw.get("tokens", [])
        yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), None)
        no_token = next((t for t in tokens if t.get("outcome", "").lower() == "no"), None)

        if not yes_token or not no_token:
            return None

        # Prices may be embedded in Gamma response, or we fetch from CLOB
        yes_price = float(yes_token.get("price", 0)) or 0.0
        no_price = float(no_token.get("price", 0)) or 0.0

        # Sanity check: prices should roughly sum to ~1 (allowing spread)
        if yes_price + no_price > 0 and not (0.8 <= yes_price + no_price <= 1.2):
            log.debug("Suspicious prices for market %s: yes=%.3f no=%.3f", raw.get("id"), yes_price, no_price)

        liquidity = float(raw.get("liquidity", 0) or 0)

        mc = MarketCondition(
            market_id=raw.get("id", ""),
            question=question,
            condition_type=ctype,
            threshold_usd=threshold,
            expiry_ts=_parse_expiry(raw.get("endDate") or raw.get("end_date_iso")),
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=yes_token.get("token_id", ""),
            no_token_id=no_token.get("token_id", ""),
            liquidity_usd=liquidity,
            raw=raw,
        )
        return mc if mc.tradeable else None
