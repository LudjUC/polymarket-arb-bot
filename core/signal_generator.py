"""
core/signal_generator.py — Orchestrates condition evaluation → pricing → signals.

Key changes:
  - Uses two-sided signal logic (BUY YES / BUY NO) from PricingEngine
  - Missed-opportunity logging rate-limited to once per market per 60s
  - Dedup window shortened to 30s for live simulation
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


@dataclass
class TradeSignal:
    signal_id: str
    market_id: str
    question: str
    target_outcome: str          # "YES" or "NO"
    token_id: str
    market_price: float          # token price we expect to pay
    implied_certainty: float
    net_edge: float
    recommended_size_usd: float
    expected_value_usd: float
    condition_notes: str
    signal_reason: str = ""      # human-readable why we're trading
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
            "generated_at_ms": self.generated_at_ms,
            "btc_price": self.btc_price_at_signal,
        }


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
        self._recent:  dict[str, float] = {}   # market_id → last signal ts
        self._dedup_window_s: float = 30.0
        self._missed:  dict[str, float] = {}   # market_id → last missed ts
        self._missed_cooldown_s: float = 60.0

    def run(self, tick: PriceTick) -> List[TradeSignal]:
        with LatencyTimer("signal_generation_ms", METRICS):
            if self._parser.needs_refresh():
                self._parser.refresh()

            markets = self._parser.get_markets()
            if not markets:
                return []

            signals: List[TradeSignal] = []
            for market in markets:
                sig = self._evaluate(market, tick)
                if sig:
                    signals.append(sig)

            signals.sort(key=lambda s: s.priority, reverse=True)
            if signals:
                METRICS.inc("signals_generated", len(signals))

            return signals

    def _evaluate(self, market: MarketCondition, tick: PriceTick) -> Optional[TradeSignal]:
        if tick.is_stale:
            return None

        # Dedup
        if time.time() - self._recent.get(market.market_id, 0) < self._dedup_window_s:
            return None

        condition: ConditionResult = self._condition.evaluate(market, tick)
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
                    LEDGER.record_missed(
                        reason=f"net_edge={analysis.net_edge:.3f} below min={CONFIG.signal.min_edge:.3f}",
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
            btc_price_at_signal=tick.price,
        )

        self._recent[market.market_id] = time.time()
        LEDGER.record_signal(sig.to_dict())

        log.info(
            "🟢 SIGNAL [%s] side=%s  entry=%.3f  impl=%.3f  edge=%.3f  "
            "size=$%.0f  EV=$%.3f\n     %s",
            sig.signal_id, analysis.target_outcome,
            analysis.market_price, analysis.implied_certainty,
            analysis.net_edge, analysis.recommended_size_usd,
            analysis.expected_value_usd,
            analysis.signal_reason,
        )

        return sig

    def update_capital(self, available: float) -> None:
        self._capital = available
