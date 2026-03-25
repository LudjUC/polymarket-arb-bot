"""
core/market_parser.py — Fetch, filter, and parse Polymarket BTC markets.

KEY CHANGE: SHORT-TERM ONLY.
We now enforce a MAX expiry window (default 72h) so we only trade markets
that can realistically resolve soon. Long-term markets ("$1M BTC before
GTA VI") are excluded entirely — their Polymarket prices do NOT react to
intra-day BTC moves, so there is no exploitable edge.

Short-term markets we want:
  "Will BTC be above $X today?"
  "Will BTC reach $X this week?"
  "Will BTC close above $X on Friday?"
"""

import json
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
    condition_type: str       # "price_above" | "price_below" | "unknown"
    threshold_usd: Optional[float]
    expiry_ts: int             # Unix timestamp; 0 = unknown
    yes_price: float
    no_price: float
    yes_token_id: str
    no_token_id: str
    liquidity_usd: float
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
    def tradeable(self) -> bool:
        cfg = CONFIG.polymarket
        tte = self.seconds_to_expiry

        # expiry_ts == 0 → unknown expiry.
        # IMPORTANT: for our short-term strategy, unknown-expiry markets are
        # EXCLUDED — they are almost always long-term markets.
        if self.expiry_ts == 0:
            return False

        # Must not be expired
        if self.is_expired:
            return False

        # Must be within our short-term window
        if tte > cfg.max_time_to_expiry_s:
            return False

        # Must not be about to expire in the next 10 minutes
        if tte < cfg.min_time_to_expiry_s:
            return False

        return (
            self.has_sufficient_liquidity
            and self.yes_price > 0
            and self.no_price > 0
        )


# ---------------------------------------------------------------------------
# Threshold parsing
# ---------------------------------------------------------------------------

_ABOVE_PATTERN = re.compile(
    r"(?:above|over|exceed|reach|hit|past|greater than|at least|>)\s*\$?([\d,]+)\s*(k|K)?",
    re.I,
)
_BELOW_PATTERN = re.compile(
    r"(?:below|under|drop(?:\s+to)?|fall(?:\s+to)?|less than|<)\s*\$?([\d,]+)\s*(k|K)?",
    re.I,
)
_BARE_DOLLAR = re.compile(r"\$\s*([\d,]+)\s*(k|K)?", re.I)


def _parse_threshold(question: str) -> tuple[Optional[float], str]:
    def _usd(digits: str, k: Optional[str]) -> float:
        return float(digits.replace(",", "")) * (1000 if k else 1)

    m = _BELOW_PATTERN.search(question)
    if m:
        try:
            return _usd(m.group(1), m.group(2)), "price_below"
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
            if 1_000 <= val <= 10_000_000:
                return val, "price_above"
        except (ValueError, IndexError):
            pass

    return None, "unknown"


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
        self._session.headers.update({"User-Agent": "polymarket-arb-bot/1.0"})
        self._cfg = CONFIG.polymarket

    def _get(self, url: str, params: Optional[Dict] = None, timeout: int = 10) -> Optional[object]:
        try:
            resp = self._session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.error("API error: %s | %s", url, exc)
            METRICS.inc("api_errors")
            return None

    def search_btc_markets(self) -> List[Dict]:
        """
        Fetch BTC markets from Gamma API.
        Uses both 'bitcoin' and 'btc' keywords and deduplicates.

        Gamma API boolean params must be strings "true"/"false".
        """
        seen: set = set()
        results: List[Dict] = []

        base = self._cfg.gamma_api_url
        for kw in ("bitcoin", "btc"):
            data = self._get(
                f"{base}/markets",
                params={"active": "true", "closed": "false", "limit": "100", "question": kw},
            )
            if not data:
                continue
            markets = data if isinstance(data, list) else data.get("markets", [])
            for m in markets:
                mid = str(m.get("id") or m.get("conditionId") or "")
                if mid and mid not in seen:
                    seen.add(mid)
                    results.append(m)

        log.info("Gamma API: %d unique BTC markets", len(results))
        return results


# ---------------------------------------------------------------------------
# Market parser
# ---------------------------------------------------------------------------

class MarketParser:

    def __init__(self) -> None:
        self._client = PolymarketClient()
        self._markets: Dict[str, MarketCondition] = {}
        self._last_refresh: float = 0.0

    def refresh(self) -> List[MarketCondition]:
        raw_list = self._client.search_btc_markets()
        parsed: List[MarketCondition] = []
        rejected_counts: Dict[str, int] = {}

        for raw in raw_list:
            mc, reason = self._parse_market(raw)
            if mc:
                self._markets[mc.market_id] = mc
                parsed.append(mc)
            else:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

        self._last_refresh = time.time()
        METRICS.gauge("active_markets", len(parsed))

        if parsed:
            log.info(
                "Loaded %d short-term BTC markets (%d rejected)",
                len(parsed), sum(rejected_counts.values()),
            )
            for mc in parsed:
                tte_h = mc.seconds_to_expiry / 3600
                log.info(
                    "  ✓ [%4.0fh] yes=%.3f liq=$%-8.0f %s",
                    tte_h, mc.yes_price, mc.liquidity_usd, mc.question[:65],
                )
        else:
            log.warning(
                "0 short-term BTC markets found. Rejection summary: %s",
                rejected_counts,
            )
            log.warning(
                "This is expected if Polymarket has no markets expiring within "
                "%.0f hours right now. Try running during high-volatility periods "
                "when daily/weekly resolution markets are active.",
                CONFIG.polymarket.max_time_to_expiry_s / 3600,
            )

        return parsed

    def get_markets(self) -> List[MarketCondition]:
        return list(self._markets.values())

    def needs_refresh(self) -> bool:
        return time.time() - self._last_refresh >= CONFIG.polymarket.market_refresh_interval_s

    # -----------------------------------------------------------------------

    def _parse_market(self, raw: Dict) -> tuple["Optional[MarketCondition]", str]:
        """Returns (MarketCondition, "") on success or (None, rejection_reason)."""
        question = raw.get("question", "")

        if not any(w in question.lower() for w in ["btc", "bitcoin"]):
            return None, "not_btc"

        threshold, ctype = _parse_threshold(question)
        if threshold is None:
            log.debug("No threshold: %r", question[:70])
            # Still continue — some valid markets have dollar signs in non-standard places

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

        # Infer missing price
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

        mc = MarketCondition(
            market_id=market_id,
            question=question,
            condition_type=ctype,
            threshold_usd=threshold,
            expiry_ts=expiry_ts,
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=str(yes_tok.get("token_id", "") or ""),
            no_token_id=str(no_tok.get("token_id",  "") or ""),
            liquidity_usd=liquidity,
            raw=raw,
        )

        if not mc.tradeable:
            # Compute the specific reason for rejection logging
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
            return None, "tradeable_check_failed"

        return mc, ""
