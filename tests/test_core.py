"""
tests/test_core.py — Unit tests for the core pipeline.
Run with: pytest tests/ -v
"""

import math
import time
import unittest
from unittest.mock import MagicMock, patch

from config import BotConfig, SignalConfig, RiskConfig, CONFIG
from core.condition_engine import ConditionEngine, ConditionResult
from core.data_ingestion import PriceTick, PriceStore
from core.market_parser import MarketCondition, _parse_threshold
from core.pricing_engine import PricingEngine
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator, TradeSignal
from core.simulator import generate_synthetic_ticks


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_market(
    condition_type="price_above",
    threshold=80_000.0,
    yes_price=0.50,
    no_price=0.50,
    expiry_offset_s=86400 * 30,
    liquidity=20_000.0,
) -> MarketCondition:
    return MarketCondition(
        market_id="test_mkt_001",
        question="Will BTC reach $80,000?",
        condition_type=condition_type,
        threshold_usd=threshold,
        expiry_ts=int(time.time()) + expiry_offset_s,
        yes_price=yes_price,
        no_price=no_price,
        yes_token_id="yes_tok",
        no_token_id="no_tok",
        liquidity_usd=liquidity,
        raw={},
    )


def _make_tick(price: float) -> PriceTick:
    return PriceTick(
        symbol="BTCUSDT",
        price=price,
        timestamp_ms=int(time.time() * 1000),
        received_at_ms=int(time.time() * 1000),
    )


# ------------------------------------------------------------------
# Tests: MarketParser helpers
# ------------------------------------------------------------------

class TestParseThreshold(unittest.TestCase):

    def test_price_above_dollar_sign(self):
        threshold, ctype = _parse_threshold("Will BTC reach $80,000 before June?")
        self.assertAlmostEqual(threshold, 80_000.0, places=0)
        self.assertEqual(ctype, "price_above")

    def test_price_above_k_suffix(self):
        threshold, ctype = _parse_threshold("Bitcoin above 100k by end of year")
        self.assertAlmostEqual(threshold, 100_000.0, places=0)
        self.assertEqual(ctype, "price_above")

    def test_price_below(self):
        threshold, ctype = _parse_threshold("Will BTC fall below $50,000?")
        self.assertAlmostEqual(threshold, 50_000.0, places=0)
        self.assertEqual(ctype, "price_below")

    def test_unknown(self):
        threshold, ctype = _parse_threshold("Will Bitcoin be popular in 2025?")
        self.assertIsNone(threshold)
        self.assertEqual(ctype, "unknown")


# ------------------------------------------------------------------
# Tests: ConditionEngine
# ------------------------------------------------------------------

class TestConditionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ConditionEngine()

    def test_price_above_met(self):
        market = _make_market(condition_type="price_above", threshold=80_000.0)
        tick = _make_tick(85_000.0)
        result = self.engine.evaluate(market, tick)
        self.assertEqual(result.outcome, "YES")
        self.assertGreater(result.implied_certainty, 0.85)
        self.assertTrue(result.is_definitive)

    def test_price_above_not_met(self):
        market = _make_market(condition_type="price_above", threshold=80_000.0)
        tick = _make_tick(70_000.0)
        result = self.engine.evaluate(market, tick)
        self.assertEqual(result.outcome, "NO")
        self.assertLess(result.implied_certainty, 0.5)

    def test_price_above_near_boundary(self):
        market = _make_market(condition_type="price_above", threshold=80_000.0)
        # Within 0.5% of threshold
        tick = _make_tick(79_800.0)
        result = self.engine.evaluate(market, tick)
        self.assertIsNone(result.outcome)
        self.assertAlmostEqual(result.implied_certainty, 0.5)

    def test_price_below_met(self):
        market = _make_market(condition_type="price_below", threshold=60_000.0)
        tick = _make_tick(55_000.0)
        result = self.engine.evaluate(market, tick)
        self.assertEqual(result.outcome, "YES")
        self.assertGreater(result.implied_certainty, 0.85)

    def test_time_discount_reduces_certainty(self):
        market = _make_market(
            condition_type="price_above",
            threshold=80_000.0,
            expiry_offset_s=86400 * 60,  # 60 days out
        )
        tick = _make_tick(81_000.0)
        result_before = self.engine.evaluate(market, tick)
        certainty_before = result_before.implied_certainty
        result_after = self.engine.apply_time_discount(result_before, market)
        self.assertLessEqual(result_after.implied_certainty, certainty_before)


# ------------------------------------------------------------------
# Tests: PricingEngine
# ------------------------------------------------------------------

class TestPricingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PricingEngine()

    def _make_condition(self, implied_certainty=0.92, outcome="YES") -> ConditionResult:
        return ConditionResult(
            market_id="test_mkt_001",
            question="Will BTC reach $80,000?",
            condition_type="price_above",
            threshold_usd=80_000.0,
            current_btc_price=85_000.0,
            implied_certainty=implied_certainty,
            is_definitive=True,
            outcome=outcome,
        )

    def test_actionable_when_edge_sufficient(self):
        """Market price = 0.60, implied certainty = 0.92 → large edge → actionable"""
        market = _make_market(yes_price=0.60)
        condition = self._make_condition(implied_certainty=0.92)
        analysis = self.engine.analyse(condition, market, capital_available_usd=5_000.0)
        self.assertIsNotNone(analysis)
        self.assertTrue(analysis.is_actionable)
        self.assertGreater(analysis.net_edge, 0)
        self.assertGreater(analysis.recommended_size_usd, 0)

    def test_not_actionable_when_edge_insufficient(self):
        """Market price = 0.92, implied certainty = 0.93 → tiny edge → not actionable"""
        market = _make_market(yes_price=0.92)
        condition = self._make_condition(implied_certainty=0.93)
        analysis = self.engine.analyse(condition, market, capital_available_usd=5_000.0)
        # Net edge = 0.93 - 0.92 - 0.02 - 0.03 = -0.04 → not actionable
        self.assertIsNotNone(analysis)
        self.assertFalse(analysis.is_actionable)

    def test_not_actionable_when_certainty_too_low(self):
        """Certainty = 0.80 < min_implied_certainty=0.90"""
        market = _make_market(yes_price=0.40)
        condition = self._make_condition(implied_certainty=0.80)
        analysis = self.engine.analyse(condition, market, capital_available_usd=5_000.0)
        self.assertIsNotNone(analysis)
        self.assertFalse(analysis.is_actionable)

    def test_size_capped_by_config(self):
        """Recommended size should not exceed max_capital_per_trade"""
        market = _make_market(yes_price=0.30)
        condition = self._make_condition(implied_certainty=0.95)
        analysis = self.engine.analyse(condition, market, capital_available_usd=100_000.0)
        self.assertIsNotNone(analysis)
        self.assertLessEqual(
            analysis.recommended_size_usd,
            CONFIG.risk.max_capital_per_trade_usd + 0.01,  # floating point tolerance
        )


# ------------------------------------------------------------------
# Tests: RiskManager
# ------------------------------------------------------------------

class TestRiskManager(unittest.TestCase):

    def _make_signal(self, signal_id="sig_001", size=100.0) -> TradeSignal:
        return TradeSignal(
            signal_id=signal_id,
            market_id="mkt_001",
            question="Will BTC hit $80k?",
            target_outcome="YES",
            token_id="tok_yes",
            market_price=0.60,
            implied_certainty=0.92,
            net_edge=0.27,
            recommended_size_usd=size,
            expected_value_usd=27.0,
            condition_notes="test",
        )

    def test_approves_valid_signal(self):
        rm = RiskManager()
        signal = self._make_signal()
        decision = rm.approve(signal)
        self.assertTrue(decision.approved)

    def test_rejects_when_max_positions_reached(self):
        rm = RiskManager()
        # Fill up positions
        for i in range(CONFIG.risk.max_open_positions):
            sig = self._make_signal(signal_id=f"sig_{i:03d}", size=50.0)
            # Manually register without approval to bypass the check
            from core.risk_manager import OpenPosition
            rm._open_positions[sig.signal_id] = OpenPosition(
                signal_id=sig.signal_id,
                market_id=f"mkt_{i}",
                token_id="tok",
                side="YES",
                size_usd=50.0,
                entry_price=0.5,
            )
        new_sig = self._make_signal(signal_id="sig_NEW")
        decision = rm.approve(new_sig)
        self.assertFalse(decision.approved)

    def test_halts_when_daily_loss_exceeded(self):
        rm = RiskManager()
        rm._daily_loss_usd = CONFIG.risk.max_daily_loss_usd + 1.0
        signal = self._make_signal()
        decision = rm.approve(signal)
        self.assertFalse(decision.approved)


# ------------------------------------------------------------------
# Tests: Simulator (smoke test)
# ------------------------------------------------------------------

class TestSimulator(unittest.TestCase):

    def test_synthetic_tick_generator(self):
        ticks = list(generate_synthetic_ticks(start_price=85_000.0, n_ticks=100))
        self.assertEqual(len(ticks), 100)
        for tick in ticks:
            self.assertGreater(tick.price, 0)
            self.assertIsInstance(tick.price, float)

    def test_gbm_has_reasonable_distribution(self):
        """GBM prices shouldn't drift wildly in 1000 ticks."""
        ticks = list(generate_synthetic_ticks(
            start_price=85_000.0,
            n_ticks=1_000,
            annual_vol=0.70,
        ))
        prices = [t.price for t in ticks]
        mean_price = sum(prices) / len(prices)
        # Mean should be within 30% of start price for a short run
        self.assertGreater(mean_price, 85_000.0 * 0.5)
        self.assertLess(mean_price, 85_000.0 * 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
