"""
core/pricing_engine.py — Edge calculation and position sizing.

Changes from v1:
  - min_edge is now tiered by market expiry (short/medium/long).
    Short (<72h):   10% net edge required
    Medium (72h–7d): 18% required
    Long (>7d):      28% required
  - Tier is read from market.expiry_tier so logic stays in one place.
  - Everything else (Kelly sizing, fee deduction, signal gating) unchanged.
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
    market_price: float          # token price we pay
    implied_certainty: float     # our honest probability estimate
    raw_edge: float              # implied_certainty - market_price
    fee_cost: float
    net_edge: float              # raw_edge - fee_cost
    kelly_fraction: float
    recommended_size_usd: float
    expected_value_usd: float
    is_actionable: bool
    expiry_tier: str = "short"   # "short" | "medium" | "long"
    signal_reason: str = ""
    notes: str = ""


def _min_edge_for_tier(tier: str) -> float:
    """Return the minimum net edge required for a given expiry tier."""
    return {
        "short":  CONFIG.signal.min_edge_short,
        "medium": CONFIG.signal.min_edge_medium,
        "long":   CONFIG.signal.min_edge_long,
    }.get(tier, CONFIG.signal.min_edge_short)


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
        Apply two-sided signal logic:

        BUY YES: condition met  AND  yes_price < buy_yes_max_price
        BUY NO:  condition not met  AND  yes_price > buy_no_min_yes_price

        min_edge is scaled by market.expiry_tier so longer-term trades
        require stronger mispricing before we act.
        """
        if condition.outcome is None:
            return None
        if not market.tradeable:
            return None

        tier = market.expiry_tier

        if condition.outcome == "YES":
            market_price = max(0.001, min(0.999, market.yes_price))
            implied_cert = condition.implied_certainty

            if market_price >= self._sig.buy_yes_max_price:
                log.debug(
                    "Skip YES [%s]: yes_price=%.3f >= buy_yes_max=%.3f  %s",
                    tier, market_price, self._sig.buy_yes_max_price,
                    market.question[:50],
                )
                return None

            reason = (
                f"{market.asset} ${condition.current_btc_price:,.0f} "
                f"≥ ${condition.threshold_usd:,.0f}  "
                f"but YES only priced at {market_price:.2f}  "
                f"[{tier} market]"
            )
            return self._compute(
                condition, market, "YES", market_price, implied_cert,
                capital_available_usd, reason, tier,
            )

        else:  # condition.outcome == "NO"
            market_price = max(0.001, min(0.999, market.no_price))
            implied_cert = 1.0 - condition.implied_certainty

            if market.yes_price <= self._sig.buy_no_min_yes_price:
                log.debug(
                    "Skip NO [%s]: yes_price=%.3f <= buy_no_min=%.3f  %s",
                    tier, market.yes_price, self._sig.buy_no_min_yes_price,
                    market.question[:50],
                )
                return None

            reason = (
                f"{market.asset} ${condition.current_btc_price:,.0f} "
                f"< ${condition.threshold_usd:,.0f}  "
                f"but YES still priced at {market.yes_price:.2f}  "
                f"→ BUY NO  [{tier} market]"
            )
            return self._compute(
                condition, market, "NO", market_price, implied_cert,
                capital_available_usd, reason, tier,
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
        tier: str,
    ) -> PricingAnalysis:

        implied_certainty = max(0.0, min(1.0, implied_certainty))

        raw_edge = implied_certainty - market_price
        fee      = self._sig.taker_fee_fraction
        net_edge = raw_edge - fee

        min_edge = _min_edge_for_tier(tier)

        # Kelly sizing
        b = max(0.001, (1.0 / market_price) - 1.0)
        p = implied_certainty
        q = 1.0 - p
        kelly_f = max(0.0, (p * b - q) / b)

        fractional_kelly = kelly_f * self._risk.kelly_fraction
        kelly_size       = fractional_kelly * self._risk.total_capital_usd
        recommended_size = min(kelly_size, self._risk.max_capital_per_trade_usd, capital_available_usd)
        recommended_size = max(0.0, recommended_size)

        ev_usd = net_edge * recommended_size

        is_actionable = (
            net_edge >= min_edge
            and recommended_size > 0
            and implied_certainty > market_price
        )

        METRICS.observe("raw_edge", raw_edge)
        METRICS.observe("net_edge", net_edge)

        log.debug(
            "[%s][%s] %s: mkt=%.3f impl=%.3f raw_edge=%.3f net=%.3f "
            "min_edge=%.3f size=$%.0f ev=$%.3f",
            tier, target, market.market_id[:10],
            market_price, implied_certainty, raw_edge, net_edge,
            min_edge, recommended_size, ev_usd,
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
            expiry_tier=tier,
            signal_reason=signal_reason,
        )
