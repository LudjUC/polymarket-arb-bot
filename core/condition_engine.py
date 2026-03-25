"""
core/condition_engine.py — Short-term BTC threshold mispricing engine.

Strategy:
  We are NOT predicting where BTC will go.
  We are detecting where it ALREADY IS versus where Polymarket THINKS it is.

Signal logic:
  BUY YES when: BTC >= threshold  AND  yes_price < buy_yes_max_price (0.70)
  BUY NO  when: BTC <  threshold  AND  yes_price > buy_no_min_yes_price (0.30)

Momentum boost:
  If BTC moved >0.5% in the last 2 minutes in the signal's direction,
  we increase our implied probability estimate slightly — the move is
  likely to be sticky for at least a few more minutes.

No fake certainty values. No instant resolution to 1.0.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from config import CONFIG
from core.data_ingestion import PriceTick
from core.market_parser import MarketCondition
from utils.logger import get_logger

log = get_logger("condition_engine")


# ---------------------------------------------------------------------------
# Price history for momentum detection
# ---------------------------------------------------------------------------

class PriceHistory:
    """Rolling window of (timestamp, price) for momentum calculation."""

    def __init__(self, window_s: float = 120.0) -> None:
        self._window_s = window_s
        self._buf: Deque[Tuple[float, float]] = deque()  # (ts, price)

    def add(self, tick: PriceTick) -> None:
        now = time.time()
        self._buf.append((now, tick.price))
        # Prune old entries
        cutoff = now - self._window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def momentum_pct(self) -> Optional[float]:
        """
        Return the price change as a fraction over the rolling window.
        Positive = price went up, negative = price went down.
        Returns None if we have fewer than 2 data points.
        """
        if len(self._buf) < 2:
            return None
        oldest_price = self._buf[0][1]
        latest_price = self._buf[-1][1]
        if oldest_price <= 0:
            return None
        return (latest_price - oldest_price) / oldest_price

    def current_price(self) -> Optional[float]:
        return self._buf[-1][1] if self._buf else None


# Global price history (shared across all condition evaluations)
PRICE_HISTORY = PriceHistory(window_s=CONFIG.signal.momentum_window_s)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    market_id: str
    question: str
    condition_type: str
    threshold_usd: Optional[float]
    current_btc_price: float
    implied_certainty: float     # 0.0–1.0, our honest estimate
    is_definitive: bool          # True when condition clearly met/failed
    outcome: Optional[str]       # "YES" | "NO" | None (near boundary / unknown)
    momentum_pct: Optional[float] = None   # BTC move over last N minutes
    momentum_boost: float = 0.0            # certainty boost from momentum
    notes: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConditionEngine:
    """
    Evaluates whether a short-term BTC threshold market is mispriced
    relative to the current live BTC price.
    """

    # Buffer zone: if BTC is within this fraction of the threshold,
    # skip the trade — too uncertain, slippage kills edge.
    BOUNDARY_FRACTION = 0.003    # 0.3% each side — tighter than before

    def evaluate(
        self,
        market: MarketCondition,
        tick: PriceTick,
    ) -> ConditionResult:
        PRICE_HISTORY.add(tick)
        price = tick.price

        if market.condition_type == "price_above" and market.threshold_usd:
            return self._eval_price_above(market, price)

        if market.condition_type == "price_below" and market.threshold_usd:
            return self._eval_price_below(market, price)

        return ConditionResult(
            market_id=market.market_id,
            question=market.question,
            condition_type=market.condition_type,
            threshold_usd=market.threshold_usd,
            current_btc_price=price,
            implied_certainty=0.5,
            is_definitive=False,
            outcome=None,
            notes="Unparseable condition type",
        )

    # -----------------------------------------------------------------------

    def _eval_price_above(self, market: MarketCondition, price: float) -> ConditionResult:
        """
        "Will BTC be above $X?"

        Our implied probability is based purely on current distance:
          - Already above by > 2%  → ~0.92 implied prob
          - Already above by 0.5–2% → ~0.80–0.92
          - Within 0.3% boundary   → skip (too uncertain)
          - Below threshold        → low prob (only trade NO if yes_price > 0.30)
        """
        threshold = market.threshold_usd
        boundary = threshold * self.BOUNDARY_FRACTION

        momentum = PRICE_HISTORY.momentum_pct()
        momentum_boost = 0.0

        if price >= threshold:
            distance_pct = (price - threshold) / threshold
            if distance_pct < self.BOUNDARY_FRACTION:
                # Too close — skip
                return self._near_boundary(market, price, momentum)

            # Base implied probability from distance
            # 0.5% above → ~0.78, 1% above → ~0.83, 2% above → ~0.90, 5% above → ~0.97
            base_certainty = 0.70 + min(distance_pct * 5.5, 0.27)
            base_certainty = min(base_certainty, 0.97)

            # Momentum boost: if BTC is still moving UP, price is stickier
            if momentum is not None and momentum > CONFIG.signal.momentum_pct_threshold:
                momentum_boost = min(momentum * 2.0, 0.05)  # up to +5%
                base_certainty = min(base_certainty + momentum_boost, 0.97)

            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_above",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=base_certainty,
                is_definitive=True,
                outcome="YES",
                momentum_pct=momentum,
                momentum_boost=momentum_boost,
                notes=(
                    f"BTC ${price:,.0f} above ${threshold:,.0f} "
                    f"(+{distance_pct:.2%})"
                    + (f" momentum={momentum:+.2%}" if momentum else "")
                ),
            )

        elif price >= threshold - boundary:
            return self._near_boundary(market, price, momentum)

        else:
            # BTC is below threshold — compute how far below
            distance_pct = (threshold - price) / threshold
            # Implied YES prob is LOW — the further below, the less likely
            # to recover within the expiry window.
            # 1% below → ~0.22 implied, 3% below → ~0.12, 5% below → ~0.07
            base_certainty = max(0.05, 0.28 - distance_pct * 4.0)

            # Momentum drag: if BTC is falling, even less likely
            if momentum is not None and momentum < -CONFIG.signal.momentum_pct_threshold:
                base_certainty = max(0.04, base_certainty + momentum * 1.5)

            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_above",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=base_certainty,
                is_definitive=False,
                outcome="NO",
                momentum_pct=momentum,
                notes=(
                    f"BTC ${price:,.0f} below ${threshold:,.0f} "
                    f"(-{distance_pct:.2%})"
                ),
            )

    def _eval_price_below(self, market: MarketCondition, price: float) -> ConditionResult:
        """Mirror of _eval_price_above for below-threshold markets."""
        threshold = market.threshold_usd
        boundary = threshold * self.BOUNDARY_FRACTION

        momentum = PRICE_HISTORY.momentum_pct()
        momentum_boost = 0.0

        if price <= threshold:
            distance_pct = (threshold - price) / threshold
            if distance_pct < self.BOUNDARY_FRACTION:
                return self._near_boundary(market, price, momentum)

            base_certainty = 0.70 + min(distance_pct * 5.5, 0.27)
            base_certainty = min(base_certainty, 0.97)

            # Momentum boost: still falling → even less likely to bounce above
            if momentum is not None and momentum < -CONFIG.signal.momentum_pct_threshold:
                momentum_boost = min(abs(momentum) * 2.0, 0.05)
                base_certainty = min(base_certainty + momentum_boost, 0.97)

            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_below",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=base_certainty,
                is_definitive=True,
                outcome="YES",
                momentum_pct=momentum,
                momentum_boost=momentum_boost,
                notes=(
                    f"BTC ${price:,.0f} below ${threshold:,.0f} "
                    f"(-{distance_pct:.2%})"
                    + (f" momentum={momentum:+.2%}" if momentum else "")
                ),
            )

        elif price <= threshold + boundary:
            return self._near_boundary(market, price, momentum)

        else:
            distance_pct = (price - threshold) / threshold
            base_certainty = max(0.05, 0.28 - distance_pct * 4.0)
            if momentum is not None and momentum > CONFIG.signal.momentum_pct_threshold:
                base_certainty = max(0.04, base_certainty - momentum * 1.5)

            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_below",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=base_certainty,
                is_definitive=False,
                outcome="NO",
                momentum_pct=momentum,
                notes=f"BTC ${price:,.0f} above threshold ${threshold:,.0f} (+{distance_pct:.2%})",
            )

    # -----------------------------------------------------------------------

    def _near_boundary(
        self, market: MarketCondition, price: float, momentum: Optional[float]
    ) -> ConditionResult:
        return ConditionResult(
            market_id=market.market_id,
            question=market.question,
            condition_type=market.condition_type,
            threshold_usd=market.threshold_usd,
            current_btc_price=price,
            implied_certainty=0.5,
            is_definitive=False,
            outcome=None,
            momentum_pct=momentum,
            notes=f"BTC ${price:,.0f} within boundary of threshold ${market.threshold_usd:,.0f}",
        )

    def apply_time_discount(
        self, result: ConditionResult, market: MarketCondition
    ) -> ConditionResult:
        """
        For SHORT-TERM markets (1–72h), apply a small time-to-expiry discount
        only when the threshold has NOT been crossed (i.e., outcome=="NO" trade).
        When BTC is clearly above/below threshold (outcome=="YES"), the
        discount is negligible for short windows — skip it to avoid over-penalising.
        """
        if not result.is_definitive:
            return result

        if market.expiry_ts == 0:
            return result

        tte_hours = max(0.0, market.seconds_to_expiry / 3600)
        if tte_hours <= 0:
            return result

        # For YES signals (condition already met): minimal discount for short TTE
        # For NO signals (condition not met): slightly more discount
        if result.outcome == "YES" and tte_hours <= 72:
            # BTC is already across — small discount, capped at 8%
            hourly_vol = 0.0075
            if market.threshold_usd and result.current_btc_price:
                distance_pct = abs(result.current_btc_price - market.threshold_usd) / market.threshold_usd
                sigma_T = hourly_vol * math.sqrt(tte_hours)
                if sigma_T > 0:
                    z = distance_pct / sigma_T
                    reversal_prob = _ncdf(-z)
                    discount = min(reversal_prob * 0.4, 0.08)
                    before = result.implied_certainty
                    result.implied_certainty = max(0.0, min(1.0, before * (1 - discount)))
                    if discount > 0.01:
                        result.notes += f" | rev_risk={discount:.2%}"

        return result


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
