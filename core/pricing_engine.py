"""
core/pricing_engine.py — Edge calculation and position sizing.

Key changes from previous version:
  - Signal logic driven by config thresholds (buy_yes_max_price, buy_no_min_yes_price)
    NOT by min_implied_certainty being artificially high.
  - Edge = (implied_certainty - market_price) after fees.
  - Only actionable when edge > min_edge (default 0.10) after fees.
  - NO fake certainty. NO instant resolution.
"""

from dataclasses import dataclass
from typing import Optional

from config import CONFIG
from core.condition_engine import ConditionResult
from core.market_parser import MarketCondition
from utils.logger import get_logger
from utils.metrics import METRICS

log = get_logger("pricing_engine")


@dataclass
class PricingAnalysis:
    market_id: str
    question: str
    target_outcome: str          # "YES" or "NO"
    market_price: float          # Polymarket token price (what we pay)
    implied_certainty: float     # Our estimate of true probability
    raw_edge: float              # implied_certainty - market_price
    fee_cost: float
    net_edge: float              # raw_edge - fee_cost
    kelly_fraction: float
    recommended_size_usd: float
    expected_value_usd: float
    is_actionable: bool
    signal_reason: str = ""      # human-readable reason for the signal
    notes: str = ""


class PricingEngine:

    def __init__(self) -> None:
        self._sig  = CONFIG.signal
        self._exit = CONFIG.exit
        self._risk = CONFIG.risk

    def analyse(
        self,
        condition: ConditionResult,
        market: MarketCondition,
        capital_available_usd: float,
    ) -> Optional[PricingAnalysis]:
        """
        Apply the two-sided signal logic:

        BUY YES: BTC >= threshold  AND  yes_price < buy_yes_max_price
        BUY NO:  BTC <  threshold  AND  yes_price > buy_no_min_yes_price

        Returns None for near-boundary / unknown conditions.
        """
        if condition.outcome is None:
            return None
        if not market.tradeable:
            return None

        if condition.outcome == "YES":
            # ── BUY YES signal ────────────────────────────────────────
            market_price = max(0.001, min(0.999, market.yes_price))
            implied_cert = condition.implied_certainty

            # Gate: market must be meaningfully underpricing YES
            if market_price >= self._sig.buy_yes_max_price:
                log.debug(
                    "Skip YES signal: yes_price=%.3f >= buy_yes_max=%.3f  %s",
                    market_price, self._sig.buy_yes_max_price, market.question[:50],
                )
                return None

            reason = (
                f"BTC ${condition.current_btc_price:,.0f} ≥ ${condition.threshold_usd:,.0f}  "
                f"but YES only priced at {market_price:.2f}"
            )
            return self._compute(
                condition, market, "YES", market_price, implied_cert,
                capital_available_usd, reason
            )

        else:  # condition.outcome == "NO"
            # ── BUY NO signal ─────────────────────────────────────────
            # We buy the NO token. Market is overpricing YES (underpricing NO).
            market_price = max(0.001, min(0.999, market.no_price))
            # implied_certainty for NO = 1 - implied_certainty_for_YES
            implied_cert = 1.0 - condition.implied_certainty

            # Gate: YES price must be irrationally high given BTC is below threshold
            if market.yes_price <= self._sig.buy_no_min_yes_price:
                log.debug(
                    "Skip NO signal: yes_price=%.3f <= buy_no_min=%.3f  %s",
                    market.yes_price, self._sig.buy_no_min_yes_price, market.question[:50],
                )
                return None

            reason = (
                f"BTC ${condition.current_btc_price:,.0f} < ${condition.threshold_usd:,.0f}  "
                f"but YES still priced at {market.yes_price:.2f}  →  BUY NO"
            )
            return self._compute(
                condition, market, "NO", market_price, implied_cert,
                capital_available_usd, reason
            )

    # -----------------------------------------------------------------------

    def _compute(
        self,
        condition: ConditionResult,
        market: MarketCondition,
        target: str,
        market_price: float,
        implied_certainty: float,
        capital_available_usd: float,
        signal_reason: str,
    ) -> PricingAnalysis:

        implied_certainty = max(0.0, min(1.0, implied_certainty))

        raw_edge = implied_certainty - market_price
        fee      = self._sig.taker_fee_fraction
        net_edge = raw_edge - fee

        # Kelly: f* = (p*b - q) / b   where b = (1/p) - 1
        b = max(0.001, (1.0 / market_price) - 1.0)
        p = implied_certainty
        q = 1.0 - p
        kelly_f = max(0.0, (p * b - q) / b)

        fractional_kelly  = kelly_f * self._risk.kelly_fraction
        kelly_size        = fractional_kelly * self._risk.total_capital_usd
        recommended_size  = min(kelly_size, self._risk.max_capital_per_trade_usd, capital_available_usd)
        recommended_size  = max(0.0, recommended_size)

        ev_usd = net_edge * recommended_size

        is_actionable = (
            net_edge >= self._sig.min_edge
            and recommended_size > 0
            and implied_certainty > market_price   # sanity
        )

        METRICS.observe("raw_edge", raw_edge)
        METRICS.observe("net_edge", net_edge)

        log.debug(
            "[%s] %s: mkt=%.3f impl=%.3f raw_edge=%.3f net=%.3f size=$%.0f ev=$%.3f",
            target, market.market_id[:10],
            market_price, implied_certainty, raw_edge, net_edge,
            recommended_size, ev_usd,
        )

        return PricingAnalysis(
            market_id=market.market_id,
            question=market.question,
            target_outcome=target,
            market_price=market_price,
            implied_certainty=implied_certainty,
            raw_edge=raw_edge,
            fee_cost=fee,
            net_edge=net_edge,
            kelly_fraction=kelly_f,
            recommended_size_usd=recommended_size,
            expected_value_usd=ev_usd,
            is_actionable=is_actionable,
            signal_reason=signal_reason,
        )
