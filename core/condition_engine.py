"""
core/condition_engine.py — Parse market questions into executable conditions
and evaluate them against live BTC price data.

Philosophy:
  We model certainty as a float 0.0–1.0.
  - 1.0 = condition is definitely met RIGHT NOW (e.g., BTC > 50k and BTC = 60k)
  - 0.0 = condition is definitely NOT met and cannot be met before expiry
  - 0.0–1.0 = uncertain / probabilistic estimate

For purely deterministic threshold conditions with live price data,
the "probability" jumps sharply once crossed, but we include a
proximity buffer so near-threshold situations don't generate false signals.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

from config import CONFIG
from core.data_ingestion import PriceTick
from core.market_parser import MarketCondition
from utils.logger import get_logger

log = get_logger("condition_engine")


# ------------------------------------------------------------------
# Evaluation result
# ------------------------------------------------------------------

@dataclass
class ConditionResult:
    market_id: str
    question: str
    condition_type: str
    threshold_usd: Optional[float]
    current_btc_price: float
    implied_certainty: float   # 0.0 – 1.0 based on real-world data
    is_definitive: bool        # True when condition is unambiguously met/failed
    outcome: Optional[str]     # "YES" | "NO" | None (uncertain)
    notes: str = ""


# ------------------------------------------------------------------
# Core engine
# ------------------------------------------------------------------

class ConditionEngine:
    """
    Evaluates MarketConditions against the current BTC price
    and returns an implied certainty estimate.
    """

    # How close to the threshold before we consider it "near boundary"
    # (expressed as a fraction of the threshold, e.g., 0.005 = 0.5%)
    NEAR_BOUNDARY_FRACTION = 0.005

    def evaluate(
        self,
        market: MarketCondition,
        tick: PriceTick,
    ) -> ConditionResult:
        """
        Evaluate a single market condition against the current price tick.
        Returns a ConditionResult with implied_certainty.
        """
        price = tick.price

        if market.condition_type == "price_above" and market.threshold_usd:
            return self._eval_price_above(market, price)

        if market.condition_type == "price_below" and market.threshold_usd:
            return self._eval_price_below(market, price)

        # Unknown condition: can't evaluate — return neutral certainty
        log.debug("Unknown condition type for market %s", market.market_id)
        return ConditionResult(
            market_id=market.market_id,
            question=market.question,
            condition_type=market.condition_type,
            threshold_usd=market.threshold_usd,
            current_btc_price=price,
            implied_certainty=0.5,
            is_definitive=False,
            outcome=None,
            notes="Condition could not be parsed",
        )

    # ------------------------------------------------------------------

    def _eval_price_above(
        self, market: MarketCondition, price: float
    ) -> ConditionResult:
        """
        "Will BTC be above $X before [date]?"
        
        Implied certainty logic:
        - Price clearly above threshold + time left → very high certainty (YES)
        - Price at or near threshold → uncertain
        - Price well below → low certainty (skew depends on time to expiry)
        
        IMPORTANT: We do NOT model future price movement here.
        We only flag situations where price has ALREADY crossed the threshold.
        For "already met" conditions, certainty approaches 1.0.
        """
        threshold = market.threshold_usd
        near_boundary = threshold * self.NEAR_BOUNDARY_FRACTION

        if price >= threshold:
            # Already above threshold → condition met right now.
            # Certainty is NOT 1.0 because price could fall back before expiry
            # on some market formulations. We use a high but not full certainty
            # unless already very far above or very close to expiry.
            distance_pct = (price - threshold) / threshold
            # The further above, the more certain (BTC would need to crash back)
            certainty = min(0.97, 0.90 + distance_pct * 5)
            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_above",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=certainty,
                is_definitive=True,
                outcome="YES",
                notes=f"BTC=${price:,.0f} > threshold=${threshold:,.0f} (Δ={distance_pct:.2%})",
            )

        elif price >= threshold - near_boundary:
            # Very close to threshold — could go either way, do not trade
            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_above",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=0.5,
                is_definitive=False,
                outcome=None,
                notes=f"BTC=${price:,.0f} near boundary (threshold=${threshold:,.0f})",
            )
        else:
            # Below threshold — we don't model upward probability here;
            # that would require a separate price prediction model.
            # Return low certainty so signal generator skips it.
            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_above",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=0.1,
                is_definitive=False,
                outcome="NO",
                notes=f"BTC=${price:,.0f} below threshold=${threshold:,.0f}",
            )

    def _eval_price_below(
        self, market: MarketCondition, price: float
    ) -> ConditionResult:
        """
        "Will BTC be below $X before [date]?"
        Mirror of _eval_price_above.
        """
        threshold = market.threshold_usd
        near_boundary = threshold * self.NEAR_BOUNDARY_FRACTION

        if price <= threshold:
            distance_pct = (threshold - price) / threshold
            certainty = min(0.97, 0.90 + distance_pct * 5)
            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_below",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=certainty,
                is_definitive=True,
                outcome="YES",
                notes=f"BTC=${price:,.0f} < threshold=${threshold:,.0f}",
            )

        elif price <= threshold + near_boundary:
            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_below",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=0.5,
                is_definitive=False,
                outcome=None,
                notes=f"BTC=${price:,.0f} near boundary",
            )
        else:
            return ConditionResult(
                market_id=market.market_id,
                question=market.question,
                condition_type="price_below",
                threshold_usd=threshold,
                current_btc_price=price,
                implied_certainty=0.1,
                is_definitive=False,
                outcome="NO",
                notes=f"BTC=${price:,.0f} above threshold=${threshold:,.0f}",
            )

    # ------------------------------------------------------------------
    # Time-to-expiry discount
    # ------------------------------------------------------------------

    def apply_time_discount(
        self, result: ConditionResult, market: MarketCondition
    ) -> ConditionResult:
        """
        For markets that haven't expired yet and the condition is met NOW,
        there is still residual uncertainty that price reverses before expiry.
        
        We apply a small discount based on:
        - Time to expiry (more time = more chance of reversal)
        - Distance from threshold (further = less reversal risk)
        
        This is a simplistic model — a real system would use BTC volatility
        estimates (e.g., realised vol) to price this properly.
        
        NOTE: This discount makes us MORE conservative (better for risk management).
        """
        if not result.is_definitive or result.implied_certainty < 0.5:
            return result

        tte_hours = market.seconds_to_expiry / 3600
        # Approximate: each hour of exposure reduces certainty slightly
        # Annualised BTC vol ~70%, hourly vol ≈ 70% / sqrt(8760) ≈ 0.75%/hour
        hourly_vol_estimate = 0.0075
        # 1σ reversal probability (very rough)
        if market.threshold_usd and result.current_btc_price:
            distance_pct = abs(result.current_btc_price - market.threshold_usd) / market.threshold_usd
            # Probability of reversal ~ N(−d/σ√T)
            sigma_T = hourly_vol_estimate * math.sqrt(tte_hours)
            if sigma_T > 0:
                z = distance_pct / sigma_T
                # CDF approximation: probability of NOT reversing
                reversal_prob = _standard_normal_cdf(-z)
                adjusted_certainty = result.implied_certainty * (1 - reversal_prob * 0.5)
                result.implied_certainty = max(0.0, min(1.0, adjusted_certainty))
                result.notes += f" | TTE={tte_hours:.1f}h discount applied"

        return result


def _standard_normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
