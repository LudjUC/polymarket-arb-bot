"""
core/signal_generator.py — Orchestrates condition evaluation → pricing → signals.

Changes from v1:
  - "No market available" mode: when no tradeable markets exist, a
    MomentumOpportunityLogger fires at most once per 60 s, logging
    BTC/ETH moves that *would* be actionable if markets existed.
    This lets you validate strategy effectiveness during dry periods.
  - ETH price feed: the generator fetches a live ETH price via REST
    fallback when ETH markets are in scope and no dedicated feed exists.
  - Dedup window preserved at 30 s.
  - Missed-opportunity log rate-limited to once per market per 60 s.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from config import CONFIG
from core.condition_engine import ConditionEngine, ConditionResult, PRICE_HISTORY
from core.data_ingestion import PriceTick
from core.market_parser import MarketCondition, MarketParser
from core.pricing_engine import PricingAnalysis, PricingEngine
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS, LatencyTimer

log = get_logger("signal_generator")


# ---------------------------------------------------------------------------
# TradeSignal (unchanged public interface)
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    signal_id: str
    market_id: str
    question: str
    target_outcome: str
    token_id: str
    market_price: float
    implied_certainty: float
    net_edge: float
    recommended_size_usd: float
    expected_value_usd: float
    condition_notes: str
    signal_reason: str = ""
    expiry_tier: str = "short"
    generated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    btc_price_at_signal: float = 0.0

    @property
    def priority(self) -> float:
        return self.expected_value_usd

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "market_id": self.market_id,
            "question": self.question,
            "target_outcome": self.target_outcome,
            "token_id": self.token_id,
            "market_price": self.market_price,
            "implied_certainty": self.implied_certainty,
            "net_edge": self.net_edge,
            "recommended_size_usd": self.recommended_size_usd,
            "expected_value_usd": self.expected_value_usd,
            "condition_notes": self.condition_notes,
            "signal_reason": self.signal_reason,
            "expiry_tier": self.expiry_tier,
            "generated_at_ms": self.generated_at_ms,
            "btc_price": self.btc_price_at_signal,
        }


# ---------------------------------------------------------------------------
# Opportunistic momentum logger (fires when no markets are available)
# ---------------------------------------------------------------------------

class MomentumOpportunityLogger:
    """
    When no tradeable Polymarket market exists, watch BTC (and ETH if
    available) for moves >= momentum.min_move_pct over a rolling window.
    Log these as hypothetical opportunities so you can gauge whether the
    strategy *would* have fired had markets been available.

    Emits at most one log line per no_market_log_interval_s.
    """

    def __init__(self) -> None:
        self._cfg = CONFIG.momentum
        self._last_log_ts: float = 0.0
        self._window_s = self._cfg.window_s
        # Rolling price buffer: list of (ts, price)
        self._btc_buf: List[tuple] = []

    def observe(self, tick: PriceTick) -> None:
        now = time.time()
        self._btc_buf.append((now, tick.price))
        # Prune old entries
        cutoff = now - self._window_s
        self._btc_buf = [(ts, p) for ts, p in self._btc_buf if ts >= cutoff]

    def maybe_log(self, tick: PriceTick) -> None:
        now = time.time()
        if now - self._last_log_ts < self._cfg.no_market_log_interval_s:
            return

        # Always emit the "waiting" message
        log.info(
            "⏳ No tradeable markets found — waiting... "
            "(BTC=$%.2f | markets will be checked again in %.0fs)",
            tick.price,
            CONFIG.polymarket.market_refresh_interval_s,
        )

        # Check for a significant momentum move
        if len(self._btc_buf) >= 2:
            oldest_price = self._btc_buf[0][1]
            if oldest_price > 0:
                move_pct = (tick.price - oldest_price) / oldest_price
                if abs(move_pct) >= self._cfg.min_move_pct:
                    direction = "▲ UP" if move_pct > 0 else "▼ DOWN"
                    log.info(
                        "📡 MOMENTUM OPPORTUNITY (no market available): "
                        "BTC moved %s %.2f%% over %.0fs "
                        "(from $%.0f → $%.0f). "
                        "This would be actionable if a short-term threshold "
                        "market existed. Consider expanding the expiry window "
                        "or checking Polymarket manually.",
                        direction,
                        abs(move_pct) * 100,
                        self._window_s,
                        oldest_price,
                        tick.price,
                    )

        self._last_log_ts = now


# ---------------------------------------------------------------------------
# ETH price fetcher (REST fallback — no dedicated WS feed for ETH yet)
# ---------------------------------------------------------------------------

_eth_price_cache: Dict[str, float] = {"price": 0.0, "ts": 0.0}
_ETH_CACHE_TTL_S = 10.0   # refresh at most every 10 seconds


def _fetch_eth_price() -> Optional[float]:
    """
    Fetch ETH/USDT spot price from Binance REST.
    Caches the result for _ETH_CACHE_TTL_S seconds to avoid hammering the API.
    Returns None on failure.
    """
    now = time.time()
    if now - _eth_price_cache["ts"] < _ETH_CACHE_TTL_S:
        cached = _eth_price_cache["price"]
        return cached if cached > 0 else None

    url = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        price = float(resp.json()["price"])
        _eth_price_cache["price"] = price
        _eth_price_cache["ts"] = now
        METRICS.gauge("eth_price_usd", price)
        return price
    except Exception as exc:
        log.debug("ETH price fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# SignalGenerator
# ---------------------------------------------------------------------------

class SignalGenerator:

    def __init__(
        self,
        market_parser: MarketParser,
        condition_engine: Optional[ConditionEngine] = None,
        pricing_engine: Optional[PricingEngine] = None,
        capital_available_usd: float = 10_000.0,
    ) -> None:
        self._parser    = market_parser
        self._condition = condition_engine or ConditionEngine()
        self._pricing   = pricing_engine or PricingEngine()
        self._capital   = capital_available_usd
        self._counter   = 0
        self._recent:  Dict[str, float] = {}   # market_id → last signal ts
        self._dedup_window_s: float = 30.0
        self._missed:  Dict[str, float] = {}   # market_id → last missed ts
        self._missed_cooldown_s: float = 60.0
        self._momentum_logger = MomentumOpportunityLogger()

    # ------------------------------------------------------------------

    def run(self, tick: PriceTick) -> List[TradeSignal]:
        with LatencyTimer("signal_generation_ms", METRICS):
            if self._parser.needs_refresh():
                self._parser.refresh()

            markets = self._parser.get_markets()

            # Always let the momentum logger observe the tick
            self._momentum_logger.observe(tick)

            if not markets:
                self._momentum_logger.maybe_log(tick)
                return []

            # Fetch ETH price once per run if any ETH markets are in scope
            eth_price: Optional[float] = None
            if any(m.asset == "ETH" for m in markets):
                eth_price = _fetch_eth_price()

            signals: List[TradeSignal] = []
            for market in markets:
                sig = self._evaluate(market, tick, eth_price)
                if sig:
                    signals.append(sig)

            signals.sort(key=lambda s: s.priority, reverse=True)
            if signals:
                METRICS.inc("signals_generated", len(signals))

            return signals

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        market: MarketCondition,
        tick: PriceTick,
        eth_price: Optional[float],
    ) -> Optional[TradeSignal]:
        if tick.is_stale:
            return None

        # Dedup
        if time.time() - self._recent.get(market.market_id, 0) < self._dedup_window_s:
            return None

        condition: ConditionResult = self._condition.evaluate(
            market, tick, eth_price=eth_price
        )
        condition = self._condition.apply_time_discount(condition, market)

        analysis: Optional[PricingAnalysis] = self._pricing.analyse(
            condition, market, self._capital
        )

        if not analysis:
            return None

        if not analysis.is_actionable:
            if analysis.raw_edge > 0:
                now = time.time()
                if now - self._missed.get(market.market_id, 0) >= self._missed_cooldown_s:
                    self._missed[market.market_id] = now
                    min_edge_for_tier = {
                        "short":  CONFIG.signal.min_edge_short,
                        "medium": CONFIG.signal.min_edge_medium,
                        "long":   CONFIG.signal.min_edge_long,
                    }.get(market.expiry_tier, CONFIG.signal.min_edge_short)
                    LEDGER.record_missed(
                        reason=(
                            f"net_edge={analysis.net_edge:.3f} below "
                            f"min={min_edge_for_tier:.3f} "
                            f"[{market.expiry_tier} tier]"
                        ),
                        signal={"market_id": market.market_id, "net_edge": analysis.net_edge},
                    )
            return None

        self._counter += 1
        token_id = (
            market.yes_token_id if analysis.target_outcome == "YES"
            else market.no_token_id
        )

        sig = TradeSignal(
            signal_id=f"sig_{self._counter:06d}",
            market_id=market.market_id,
            question=market.question,
            target_outcome=analysis.target_outcome,
            token_id=token_id,
            market_price=analysis.market_price,
            implied_certainty=analysis.implied_certainty,
            net_edge=analysis.net_edge,
            recommended_size_usd=analysis.recommended_size_usd,
            expected_value_usd=analysis.expected_value_usd,
            condition_notes=condition.notes,
            signal_reason=analysis.signal_reason,
            expiry_tier=analysis.expiry_tier,
            btc_price_at_signal=tick.price,
        )

        self._recent[market.market_id] = time.time()
        LEDGER.record_signal(sig.to_dict())

        log.info(
            "🟢 SIGNAL [%s][%s] side=%s  entry=%.3f  impl=%.3f  "
            "edge=%.3f  size=$%.0f  EV=$%.3f\n     %s",
            sig.signal_id, analysis.expiry_tier, analysis.target_outcome,
            analysis.market_price, analysis.implied_certainty,
            analysis.net_edge, analysis.recommended_size_usd,
            analysis.expected_value_usd,
            analysis.signal_reason,
        )

        return sig

    def update_capital(self, available: float) -> None:
        self._capital = available
