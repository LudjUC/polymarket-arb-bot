"""
core/pricing_engine.py — Compare Polymarket prices vs implied certainty,
compute edge, expected value, and Kelly-optimal position size.

Key concepts:
  - Market price (p): Polymarket's current YES token price
  - Implied certainty (q): Our estimate of the true probability from real-world data
  - Edge: q - p  (positive = market is underpricing the outcome)
  - Net edge: edge - fees - safety_margin
  - EV: net_edge × position_size
  - Kelly size: f* = (q × b - (1-q)) / b  where b = (1/p) - 1
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
    target_outcome: str         # "YES" or "NO"
    market_price: float         # Polymarket token price
    implied_certainty: float    # Our estimate
    raw_edge: float             # implied_certainty - market_price
    fee_cost: float             # estimated round-trip fee fraction
    safety_margin: float        # configured safety buffer
    net_edge: float             # raw_edge - fee_cost - safety_margin
    kelly_fraction: float       # raw Kelly fraction (before fractional scaling)
    recommended_size_usd: float # Dollar size after fractional Kelly + capital cap
    expected_value_usd: float   # EV = net_edge × recommended_size_usd
    is_actionable: bool         # net_edge > 0 AND implied_certainty > min_certainty
    notes: str = ""


class PricingEngine:
    """
    Computes edge and position sizing for a given condition result.
    """

    def __init__(self) -> None:
        self._sig_cfg = CONFIG.signal
        self._risk_cfg = CONFIG.risk

    # ------------------------------------------------------------------

    def analyse(
        self,
        condition: ConditionResult,
        market: MarketCondition,
        capital_available_usd: float,
    ) -> Optional[PricingAnalysis]:
        """
        Compute edge and sizing for a single market.
        Returns None if the condition is indeterminate or market is untradeable.
        """
        if condition.outcome is None:
            return None  # Can't trade uncertain conditions
        if not market.tradeable:
            return None

        # Determine which side to trade
        if condition.outcome == "YES":
            target = "YES"
            market_price = market.yes_price
            token_id = market.yes_token_id
        else:
            # If condition says NO, trade the NO token
            target = "NO"
            market_price = market.no_price
            token_id = market.no_token_id
            # Flip implied certainty to NO side
            condition_certainty = 1.0 - condition.implied_certainty
            # Use adjusted certainty for NO side
            implied_certainty = condition_certainty
            return self._compute(
                condition, market, target, market_price, implied_certainty,
                capital_available_usd
            )

        return self._compute(
            condition, market, target, market_price,
            condition.implied_certainty, capital_available_usd
        )

    # ------------------------------------------------------------------

    def _compute(
        self,
        condition: ConditionResult,
        market: MarketCondition,
        target: str,
        market_price: float,
        implied_certainty: float,
        capital_available_usd: float,
    ) -> PricingAnalysis:
        sig = self._sig_cfg
        risk = self._risk_cfg

        # Guard against degenerate prices
        market_price = max(0.001, min(0.999, market_price))
        implied_certainty = max(0.0, min(1.0, implied_certainty))

        raw_edge = implied_certainty - market_price
        fee_cost = sig.taker_fee_fraction
        net_edge = raw_edge - fee_cost - sig.safety_margin

        # Kelly criterion: f* = (p*b - q) / b  where b = odds paid = (1-p)/p
        # Here: b = (1/market_price) - 1
        b = (1.0 / market_price) - 1.0  # payout ratio (profit per unit staked)
        p = implied_certainty
        q = 1.0 - implied_certainty
        if b > 0:
            kelly_f = (p * b - q) / b
        else:
            kelly_f = 0.0

        kelly_f = max(0.0, kelly_f)  # never short via Kelly

        # Apply fractional Kelly for safety
        fractional_kelly = kelly_f * risk.kelly_fraction

        # Position size: min of (fractional Kelly × total capital, max per trade, available)
        kelly_size = fractional_kelly * risk.total_capital_usd
        recommended_size = min(kelly_size, risk.max_capital_per_trade_usd, capital_available_usd)
        recommended_size = max(0.0, recommended_size)

        ev_usd = net_edge * recommended_size

        is_actionable = (
            net_edge > 0
            and implied_certainty >= sig.min_implied_certainty
            and recommended_size > 0
            and market_price < implied_certainty  # sanity check
        )

        METRICS.observe("raw_edge", raw_edge)
        METRICS.observe("net_edge", net_edge)

        notes = (
            f"price={market_price:.3f} certainty={implied_certainty:.3f} "
            f"raw_edge={raw_edge:.3f} net_edge={net_edge:.3f} "
            f"kelly={kelly_f:.3f} size=${recommended_size:.2f}"
        )

        log.debug(
            "Pricing analysis for %s [%s]: %s",
            market.market_id, target, notes
        )

        return PricingAnalysis(
            market_id=market.market_id,
            question=market.question,
            target_outcome=target,
            market_price=market_price,
            implied_certainty=implied_certainty,
            raw_edge=raw_edge,
            fee_cost=fee_cost,
            safety_margin=sig.safety_margin,
            net_edge=net_edge,
            kelly_fraction=kelly_f,
            recommended_size_usd=recommended_size,
            expected_value_usd=ev_usd,
            is_actionable=is_actionable,
            notes=notes,
        )
