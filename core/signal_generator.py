"""
core/signal_generator.py — Orchestrates the full pipeline:
  PriceTick → ConditionEngine → PricingEngine → ranked TradeSignals

Only emits signals that:
  1. Pass the net_edge threshold
  2. Pass the min implied certainty threshold
  3. Pass risk manager checks
  4. Are not near market expiry or illiquid
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import CONFIG
from core.condition_engine import ConditionEngine, ConditionResult
from core.data_ingestion import PriceTick
from core.market_parser import MarketCondition, MarketParser
from core.pricing_engine import PricingAnalysis, PricingEngine
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS, LatencyTimer

log = get_logger("signal_generator")


# ------------------------------------------------------------------
# Signal model
# ------------------------------------------------------------------

@dataclass
class TradeSignal:
    signal_id: str
    market_id: str
    question: str
    target_outcome: str          # "YES" or "NO"
    token_id: str                # CLOB token to buy
    market_price: float
    implied_certainty: float
    net_edge: float
    recommended_size_usd: float
    expected_value_usd: float
    condition_notes: str
    generated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    btc_price_at_signal: float = 0.0

    @property
    def priority(self) -> float:
        """Higher = more attractive. Used for ranking."""
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
            "generated_at_ms": self.generated_at_ms,
            "btc_price": self.btc_price_at_signal,
        }


# ------------------------------------------------------------------
# Signal generator
# ------------------------------------------------------------------

class SignalGenerator:
    """
    Wires together market parsing, condition evaluation, and pricing.
    Called on every significant price tick (or periodic timer).
    """

    def __init__(
        self,
        market_parser: MarketParser,
        condition_engine: Optional[ConditionEngine] = None,
        pricing_engine: Optional[PricingEngine] = None,
        capital_available_usd: float = 10_000.0,
    ) -> None:
        self._parser = market_parser
        self._condition_engine = condition_engine or ConditionEngine()
        self._pricing_engine = pricing_engine or PricingEngine()
        self._capital_available = capital_available_usd
        self._signal_counter = 0
        # Dedup: don't re-emit the same signal within N seconds
        self._recent_signals: dict[str, float] = {}   # market_id → last signal ts
        self._dedup_window_s: float = 60.0
        # Missed-opportunity cooldown: only write to ledger once per market per window
        self._missed_cooldown: dict[str, float] = {}  # market_id → last missed ts
        self._missed_cooldown_s: float = 60.0

    # ------------------------------------------------------------------

    def run(self, tick: PriceTick) -> List[TradeSignal]:
        """
        Main entry point. Given a price tick, evaluate all known markets
        and return a ranked list of trade signals.
        """
        with LatencyTimer("signal_generation_ms", METRICS):
            if self._parser.needs_refresh():
                self._parser.refresh()

            markets = self._parser.get_markets()
            if not markets:
                return []

            signals: List[TradeSignal] = []

            for market in markets:
                signal = self._evaluate_market(market, tick)
                if signal:
                    signals.append(signal)

            # Rank by EV descending
            signals.sort(key=lambda s: s.priority, reverse=True)

            if signals:
                METRICS.inc("signals_generated", len(signals))
                log.info("Generated %d signals at BTC=$%.2f", len(signals), tick.price)

            return signals

    # ------------------------------------------------------------------

    def _evaluate_market(
        self, market: MarketCondition, tick: PriceTick
    ) -> Optional[TradeSignal]:
        # Skip if stale data
        if tick.is_stale:
            log.warning("Stale price tick (%dms old) — skipping signal", tick.age_ms)
            METRICS.inc("signals_skipped_stale_data")
            return None

        # Dedup check
        last_seen = self._recent_signals.get(market.market_id, 0)
        if time.time() - last_seen < self._dedup_window_s:
            return None

        # --- Condition evaluation
        condition: ConditionResult = self._condition_engine.evaluate(market, tick)
        condition = self._condition_engine.apply_time_discount(condition, market)

        # --- Pricing analysis
        analysis: Optional[PricingAnalysis] = self._pricing_engine.analyse(
            condition, market, self._capital_available
        )

        if not analysis:
            return None

        if not analysis.is_actionable:
            # Rate-limit missed-opportunity ledger writes to once per market per window
            # to prevent per-tick log spam during live simulation.
            if analysis.raw_edge > 0:
                now = time.time()
                last_missed = self._missed_cooldown.get(market.market_id, 0)
                if now - last_missed >= self._missed_cooldown_s:
                    self._missed_cooldown[market.market_id] = now
                    LEDGER.record_missed(
                        reason=f"net_edge={analysis.net_edge:.3f} below threshold or certainty too low",
                        signal={"market_id": market.market_id, "net_edge": analysis.net_edge},
                    )
            return None

        # --- Build signal
        self._signal_counter += 1
        token_id = (
            market.yes_token_id
            if analysis.target_outcome == "YES"
            else market.no_token_id
        )

        signal = TradeSignal(
            signal_id=f"sig_{self._signal_counter:06d}",
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
            btc_price_at_signal=tick.price,
        )

        self._recent_signals[market.market_id] = time.time()
        LEDGER.record_signal(signal.to_dict())

        log.info(
            "🟢 SIGNAL [%s] market=%s side=%s price=%.3f certainty=%.3f "
            "net_edge=%.3f size=$%.2f EV=$%.4f",
            signal.signal_id,
            market.market_id[:12],
            analysis.target_outcome,
            analysis.market_price,
            analysis.implied_certainty,
            analysis.net_edge,
            analysis.recommended_size_usd,
            analysis.expected_value_usd,
        )

        return signal

    def update_capital(self, available: float) -> None:
        """Called by risk manager after each trade to update available capital."""
        self._capital_available = available
