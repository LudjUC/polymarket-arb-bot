"""
core/market_parser.py — Fetch, filter, and parse Polymarket BTC/ETH markets.

KEY CHANGES from v1:
  1. Extended expiry window: up to 14 days (was 72h).
     Markets are tiered by expiry; the pricing engine scales min_edge
     accordingly so longer-term trades require stronger mispricing.

  2. ETH fallback: if fewer than N BTC markets pass filters, we also
     search for ETH markets. The asset is tagged on each MarketCondition
     so downstream modules can adjust their data feeds if needed.

  3. Richer condition parsing: in addition to "above / below $X" we
     now recognise range markets and generic "directional" markets
     (e.g. "reach $X", "end above $X on Friday").

  4. "No-market" rate-limiting: repeated warnings are suppressed;
     a single message is emitted at most once per 60 seconds.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from config import CONFIG
from utils.logger import get_logger
from utils.metrics import METRICS

log = get_logger("market_parser")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MarketToken:
    token_id: str
    outcome: str
    price: float
    size_available: float


@dataclass
class MarketCondition:
    market_id: str
    question: str
    # Extended condition types:
    #   "price_above" | "price_below" | "price_range" | "directional" | "unknown"
    condition_type: str
    threshold_usd: Optional[float]
    threshold_upper_usd: Optional[float]   # upper bound for range markets
    expiry_ts: int                         # Unix timestamp; 0 = unknown
    yes_price: float
    no_price: float
    yes_token_id: str
    no_token_id: str
    liquidity_usd: float
    asset: str                             # "BTC" | "ETH"
    raw: Dict

    @property
    def seconds_to_expiry(self) -> float:
        return self.expiry_ts - time.time()

    @property
    def is_expired(self) -> bool:
        return self.expiry_ts != 0 and self.seconds_to_expiry <= 0

    @property
    def has_sufficient_liquidity(self) -> bool:
        return self.liquidity_usd >= CONFIG.polymarket.min_market_liquidity_usd

    @property
    def expiry_tier(self) -> str:
        """Return 'short' | 'medium' | 'long' based on time to expiry."""
        tte = self.seconds_to_expiry
        if tte <= CONFIG.polymarket.tier_short_s:
            return "short"
        if tte <= CONFIG.polymarket.tier_medium_s:
            return "medium"
        return "long"

    @property
    def tradeable(self) -> bool:
        cfg = CONFIG.polymarket
        tte = self.seconds_to_expiry

        if self.expiry_ts == 0:
            return False
        if self.is_expired:
            return False
        if tte > cfg.max_time_to_expiry_s:
            return False
        if tte < cfg.min_time_to_expiry_s:
            return False
        return (
            self.has_sufficient_liquidity
            and self.yes_price > 0
            and self.no_price > 0
            and self.condition_type != "unknown"
        )


# ---------------------------------------------------------------------------
# Threshold / condition parsing
# ---------------------------------------------------------------------------

# "above / over / exceed / reach / hit / > / at least" patterns
_ABOVE_PATTERN = re.compile(
    r"(?:above|over|exceed|reach|hit|past|greater than|at least|surpass|>)\s*"
    r"\$?([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.I,
)
# "below / under / drop / fall / less than / <" patterns
_BELOW_PATTERN = re.compile(
    r"(?:below|under|drop(?:\s+to)?|fall(?:\s+to)?|less than|dip(?:\s+below)?|<)\s*"
    r"\$?([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.I,
)
# "end above / close above / finish above $X" → treated as price_above
_CLOSE_ABOVE_PATTERN = re.compile(
    r"(?:end|close|finish|settle)\s+above\s+\$?([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.I,
)
# "end below / close below $X" → price_below
_CLOSE_BELOW_PATTERN = re.compile(
    r"(?:end|close|finish|settle)\s+below\s+\$?([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.I,
)
# Range: "between $X and $Y"
_RANGE_PATTERN = re.compile(
    r"between\s+\$?([\d,]+(?:\.\d+)?)\s*(k|K)?\s+and\s+\$?([\d,]+(?:\.\d+)?)\s*(k|K)?",
    re.I,
)
# Bare dollar sign fallback
_BARE_DOLLAR = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|K)?", re.I)


def _usd(digits: str, k: Optional[str]) -> float:
    return float(digits.replace(",", "")) * (1000.0 if k else 1.0)


def _parse_threshold(question: str) -> Tuple[Optional[float], str]:
    """
    Returns (threshold_usd, condition_type).
    condition_type is one of: "price_above" | "price_below" | "unknown"
    (Range markets use parse_full_condition instead.)
    """
    m = _CLOSE_BELOW_PATTERN.search(question)
    if m:
        try:
            return _usd(m.group(1), m.group(2)), "price_below"
        except (ValueError, IndexError):
            pass

    m = _BELOW_PATTERN.search(question)
    if m:
        try:
            return _usd(m.group(1), m.group(2)), "price_below"
        except (ValueError, IndexError):
            pass

    m = _CLOSE_ABOVE_PATTERN.search(question)
    if m:
        try:
            return _usd(m.group(1), m.group(2)), "price_above"
        except (ValueError, IndexError):
            pass

    m = _ABOVE_PATTERN.search(question)
    if m:
        try:
            return _usd(m.group(1), m.group(2)), "price_above"
        except (ValueError, IndexError):
            pass

    m = _BARE_DOLLAR.search(question)
    if m:
        try:
            val = _usd(m.group(1), m.group(2))
            if 100 <= val <= 10_000_000:
                return val, "price_above"
        except (ValueError, IndexError):
            pass

    return None, "unknown"


def _parse_full_condition(
    question: str,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extended parser that also handles range markets.
    Returns (threshold_lower, threshold_upper, condition_type).
    threshold_upper is None for non-range markets.
    """
    # Range check first
    m = _RANGE_PATTERN.search(question)
    if m:
        try:
            lo = _usd(m.group(1), m.group(2))
            hi = _usd(m.group(3), m.group(4))
            if lo < hi:
                return lo, hi, "price_range"
        except (ValueError, IndexError):
            pass

    threshold, ctype = _parse_threshold(question)
    return threshold, None, ctype


def _parse_expiry(val: Optional[str]) -> int:
    if not val:
        return 0
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class PolymarketClient:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-arb-bot/2.0"})
        self._cfg = CONFIG.polymarket

    def _get(
        self, url: str, params: Optional[Dict] = None, timeout: int = 10
    ) -> Optional[object]:
        try:
            resp = self._session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.error("API error: %s | %s", url, exc)
            METRICS.inc("api_errors")
            return None

    def search_markets(self, keywords: List[str]) -> List[Dict]:
        """
        Fetch markets for a list of keywords, deduplicating by market ID.
        """
        seen: set = set()
        results: List[Dict] = []
        base = self._cfg.gamma_api_url

        for kw in keywords:
            data = self._get(
                f"{base}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": "100",
                    "question": kw,
                },
            )
            if not data:
                continue
            markets = data if isinstance(data, list) else data.get("markets", [])
            for m in markets:
                mid = str(m.get("id") or m.get("conditionId") or "")
                if mid and mid not in seen:
                    seen.add(mid)
                    results.append(m)

        return results


# ---------------------------------------------------------------------------
# Market parser
# ---------------------------------------------------------------------------

class MarketParser:

    def __init__(self) -> None:
        self._client = PolymarketClient()
        self._markets: Dict[str, MarketCondition] = {}
        self._last_refresh: float = 0.0
        self._last_no_market_warn_ts: float = 0.0  # rate-limit "no markets" log

    def refresh(self) -> List[MarketCondition]:
        cfg = CONFIG.polymarket

        # ── Step 1: always fetch BTC markets ─────────────────────────
        btc_raw = self._client.search_markets(cfg.primary_keywords)
        log.info("Gamma API: %d unique BTC markets returned", len(btc_raw))

        btc_markets, btc_rejected = self._parse_batch(btc_raw, asset="BTC")

        # ── Step 2: ETH fallback ──────────────────────────────────────
        eth_markets: List[MarketCondition] = []
        if (
            cfg.allow_eth_fallback
            and len(btc_markets) < cfg.eth_fallback_min_btc_markets
        ):
            log.info(
                "Only %d BTC markets passed filters — fetching ETH fallback markets",
                len(btc_markets),
            )
            eth_raw = self._client.search_markets(cfg.fallback_keywords)
            log.info("Gamma API: %d unique ETH markets returned", len(eth_raw))
            eth_markets, eth_rejected = self._parse_batch(eth_raw, asset="ETH")
            log.info(
                "ETH markets: %d passed, %d rejected",
                len(eth_markets), sum(eth_rejected.values()),
            )

        all_markets = btc_markets + eth_markets

        # Rebuild internal dict
        self._markets = {m.market_id: m for m in all_markets}
        self._last_refresh = time.time()
        METRICS.gauge("active_markets", len(all_markets))

        if all_markets:
            log.info(
                "Loaded %d tradeable markets (%d BTC, %d ETH)",
                len(all_markets), len(btc_markets), len(eth_markets),
            )
            for mc in sorted(all_markets, key=lambda m: m.seconds_to_expiry):
                tte_h = mc.seconds_to_expiry / 3600
                log.info(
                    "  ✓ [%s][%5.1fh] yes=%.3f liq=$%-8.0f %s",
                    mc.asset, tte_h, mc.yes_price, mc.liquidity_usd,
                    mc.question[:60],
                )
        else:
            now = time.time()
            if now - self._last_no_market_warn_ts >= 60.0:
                self._last_no_market_warn_ts = now
                log.warning(
                    "No tradeable markets found (BTC rejected: %s). "
                    "Extended window is active (%.0fd). "
                    "Possible causes: Polymarket API schema change, all markets "
                    "expired, or liquidity below $%.0f threshold.",
                    btc_rejected,
                    cfg.max_time_to_expiry_s / 86400,
                    cfg.min_market_liquidity_usd,
                )

        return all_markets

    def get_markets(self) -> List[MarketCondition]:
        return list(self._markets.values())

    def needs_refresh(self) -> bool:
        return time.time() - self._last_refresh >= CONFIG.polymarket.market_refresh_interval_s

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_batch(
        self, raw_list: List[Dict], asset: str
    ) -> Tuple[List[MarketCondition], Dict[str, int]]:
        parsed: List[MarketCondition] = []
        rejected: Dict[str, int] = {}

        # Determine which keywords identify this asset
        if asset == "BTC":
            asset_kws = CONFIG.polymarket.primary_keywords
        else:
            asset_kws = CONFIG.polymarket.fallback_keywords

        for raw in raw_list:
            mc, reason = self._parse_market(raw, asset_kws, asset)
            if mc:
                parsed.append(mc)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1

        return parsed, rejected

    def _parse_market(
        self, raw: Dict, asset_keywords: List[str], asset: str
    ) -> Tuple["Optional[MarketCondition]", str]:
        """Returns (MarketCondition, '') on success or (None, rejection_reason)."""
        question = raw.get("question", "")

        if not any(w in question.lower() for w in asset_keywords):
            return None, "not_target_asset"

        market_id = str(raw.get("id") or raw.get("conditionId") or "")
        if not market_id:
            return None, "no_id"

        # ── Tokens ────────────────────────────────────────────────────
        tokens_raw = raw.get("tokens", [])
        if isinstance(tokens_raw, str):
            try:
                tokens_raw = json.loads(tokens_raw)
            except Exception:
                tokens_raw = []
        tokens: List[Dict] = tokens_raw if isinstance(tokens_raw, list) else []

        yes_tok = next((t for t in tokens if str(t.get("outcome", "")).lower() == "yes"), None)
        no_tok  = next((t for t in tokens if str(t.get("outcome", "")).lower() == "no"),  None)

        if (not yes_tok or not no_tok) and len(tokens) == 2:
            yes_tok, no_tok = tokens[0], tokens[1]

        outcomes_raw = raw.get("outcomes", "[]")
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = json.loads(outcomes_raw)
            except Exception:
                outcomes_raw = []
        if (not yes_tok or not no_tok) and isinstance(outcomes_raw, list) and len(outcomes_raw) >= 2:
            yes_tok = {"outcome": outcomes_raw[0], "token_id": "", "price": 0}
            no_tok  = {"outcome": outcomes_raw[1], "token_id": "", "price": 0}

        if not yes_tok or not no_tok:
            return None, "no_tokens"

        # ── Prices ────────────────────────────────────────────────────
        op_raw = raw.get("outcomePrices", "[]")
        if isinstance(op_raw, str):
            try:
                op = [float(p) for p in json.loads(op_raw)]
            except Exception:
                op = []
        elif isinstance(op_raw, list):
            try:
                op = [float(p) for p in op_raw]
            except Exception:
                op = []
        else:
            op = []

        yes_price = op[0] if len(op) >= 1 else float(yes_tok.get("price", 0) or 0)
        no_price  = op[1] if len(op) >= 2 else float(no_tok.get("price",  0) or 0)

        if yes_price == 0.0:
            yes_price = float(yes_tok.get("price", 0) or 0)
        if no_price == 0.0:
            no_price = float(no_tok.get("price", 0) or 0)

        if yes_price > 0 and no_price == 0.0:
            no_price = round(1.0 - yes_price, 4)
        elif no_price > 0 and yes_price == 0.0:
            yes_price = round(1.0 - no_price, 4)

        # ── Liquidity ─────────────────────────────────────────────────
        try:
            liquidity = float(raw.get("liquidity", 0) or raw.get("volume", 0) or 0)
        except (TypeError, ValueError):
            liquidity = 0.0

        # ── Expiry ────────────────────────────────────────────────────
        expiry_ts = _parse_expiry(
            raw.get("endDate") or raw.get("end_date_iso") or
            raw.get("endDateIso") or raw.get("gameStartTime")
        )

        # ── Condition parsing (extended) ──────────────────────────────
        threshold, threshold_upper, ctype = _parse_full_condition(question)

        mc = MarketCondition(
            market_id=market_id,
            question=question,
            condition_type=ctype,
            threshold_usd=threshold,
            threshold_upper_usd=threshold_upper,
            expiry_ts=expiry_ts,
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=str(yes_tok.get("token_id", "") or ""),
            no_token_id=str(no_tok.get("token_id",  "") or ""),
            liquidity_usd=liquidity,
            asset=asset,
            raw=raw,
        )

        if not mc.tradeable:
            if mc.expiry_ts == 0:
                return None, "no_expiry"
            if mc.is_expired:
                return None, "expired"
            tte = mc.seconds_to_expiry
            if tte > CONFIG.polymarket.max_time_to_expiry_s:
                return None, f"too_long_term({tte/3600:.0f}h)"
            if tte < CONFIG.polymarket.min_time_to_expiry_s:
                return None, "expiry_too_soon"
            if not mc.has_sufficient_liquidity:
                return None, f"low_liq(${liquidity:.0f})"
            if yes_price <= 0 or no_price <= 0:
                return None, "zero_price"
            if mc.condition_type == "unknown":
                return None, "unparseable_condition"
            return None, "tradeable_check_failed"

        return mc, ""
