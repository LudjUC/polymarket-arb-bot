"""
core/simulator.py — Historical simulation engine.

Replays a CSV of (timestamp_ms, price) ticks through the full pipeline
and reports how many opportunities would have been detected and their
aggregate EV, slippage, and outcome distribution.

CSV format expected:
  timestamp_ms,price
  1700000000000,42500.50
  1700000001000,42501.20
  ...

Can also generate synthetic BTC price data for testing when no CSV is available.
"""

import csv
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

from config import CONFIG
from core.condition_engine import ConditionEngine
from core.data_ingestion import PriceTick, PriceStore
from core.market_parser import MarketCondition, MarketParser
from core.pricing_engine import PricingEngine
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator, TradeSignal
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS

log = get_logger("simulator")


# ------------------------------------------------------------------
# Simulation result
# ------------------------------------------------------------------

@dataclass
class SimulationResult:
    total_ticks: int = 0
    total_signals: int = 0
    actionable_signals: int = 0
    total_ev_usd: float = 0.0
    avg_net_edge: float = 0.0
    max_net_edge: float = 0.0
    win_rate_estimate: float = 0.0
    missed_opportunities: int = 0
    runtime_s: float = 0.0
    signals: List[dict] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("       SIMULATION RESULTS")
        print("=" * 60)
        print(f"  Ticks processed:      {self.total_ticks:,}")
        print(f"  Signals generated:    {self.total_signals:,}")
        print(f"  Actionable signals:   {self.actionable_signals:,}")
        print(f"  Missed opportunities: {self.missed_opportunities:,}")
        print(f"  Total EV (USD):       ${self.total_ev_usd:,.4f}")
        print(f"  Avg net edge:         {self.avg_net_edge:.4f}")
        print(f"  Max net edge:         {self.max_net_edge:.4f}")
        print(f"  Win rate estimate:    {self.win_rate_estimate:.1%}")
        print(f"  Runtime:              {self.runtime_s:.2f}s")
        print("=" * 60 + "\n")


# ------------------------------------------------------------------
# Tick generators
# ------------------------------------------------------------------

def ticks_from_csv(path: str) -> Generator[PriceTick, None, None]:
    """Yield PriceTick objects from a CSV file."""
    if not os.path.exists(path):
        log.error("Historical data file not found: %s", path)
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                yield PriceTick(
                    symbol="BTCUSDT",
                    price=float(row["price"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    received_at_ms=int(row["timestamp_ms"]),
                )
            except (KeyError, ValueError) as exc:
                log.warning("Skipping malformed CSV row: %s | %s", row, exc)


def generate_synthetic_ticks(
    start_price: float = 85_000.0,
    n_ticks: int = 10_000,
    tick_interval_ms: int = 1_000,
    annual_vol: float = 0.70,
    drift: float = 0.0,
) -> Generator[PriceTick, None, None]:
    """
    Generate synthetic BTC prices via Geometric Brownian Motion.
    Useful for testing when no historical data is available.
    
    annual_vol=0.70 → daily_vol ≈ 4.4%, hourly_vol ≈ 1.1%
    """
    dt = tick_interval_ms / 1000 / (365 * 24 * 3600)  # fraction of year
    sigma = annual_vol
    mu = drift

    price = start_price
    ts = int(time.time() * 1000)

    log.info("Generating %d synthetic BTC ticks from $%.2f", n_ticks, start_price)

    for _ in range(n_ticks):
        z = random.gauss(0, 1)
        price *= math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        yield PriceTick(
            symbol="BTCUSDT",
            price=round(price, 2),
            timestamp_ms=ts,
            received_at_ms=ts,
        )
        ts += tick_interval_ms


# ------------------------------------------------------------------
# Main simulator
# ------------------------------------------------------------------

class Simulator:
    """
    Runs the full signal pipeline against historical or synthetic data.
    Does NOT place real orders. Tracks hypothetical P&L.
    """

    def __init__(
        self,
        markets: Optional[List[MarketCondition]] = None,
    ) -> None:
        self._markets = markets or []
        # Build a mock market parser that uses pre-loaded markets
        self._parser = _MockMarketParser(self._markets)
        self._gen = SignalGenerator(
            market_parser=self._parser,
            capital_available_usd=CONFIG.risk.total_capital_usd,
        )
        self._risk = RiskManager()

    def add_market(self, market: MarketCondition) -> None:
        self._markets.append(market)

    def run(
        self,
        tick_source: Optional[Generator[PriceTick, None, None]] = None,
        max_ticks: Optional[int] = None,
        playback_delay_s: float = 0.0,
    ) -> SimulationResult:
        """
        Run simulation.
        tick_source: generator of PriceTick. Defaults to synthetic.
        max_ticks: limit for fast runs.
        """
        if tick_source is None:
            tick_source = generate_synthetic_ticks()

        result = SimulationResult()
        net_edges: List[float] = []
        start_time = time.time()

        for tick in tick_source:
            if max_ticks and result.total_ticks >= max_ticks:
                break

            result.total_ticks += 1

            if playback_delay_s > 0:
                time.sleep(playback_delay_s)

            signals = self._gen.run(tick)
            result.total_signals += len(signals)

            for sig in signals:
                result.actionable_signals += 1
                result.total_ev_usd += sig.expected_value_usd
                net_edges.append(sig.net_edge)
                result.signals.append(sig.to_dict())

        ledger_summary = LEDGER.summary()
        result.missed_opportunities = ledger_summary["total_signals"] - result.actionable_signals
        result.runtime_s = time.time() - start_time

        if net_edges:
            result.avg_net_edge = sum(net_edges) / len(net_edges)
            result.max_net_edge = max(net_edges)
            # Rough win rate: fraction of signals where implied_certainty > 0.5
            winning = [s for s in result.signals if s.get("implied_certainty", 0) > 0.5]
            result.win_rate_estimate = len(winning) / len(result.signals)

        return result


# ------------------------------------------------------------------
# Mock market parser for simulation
# ------------------------------------------------------------------

class _MockMarketParser:
    """Wraps a fixed list of MarketCondition objects for simulation."""

    def __init__(self, markets: List[MarketCondition]) -> None:
        self._markets = markets

    def needs_refresh(self) -> bool:
        return False

    def refresh(self) -> List[MarketCondition]:
        return self._markets

    def get_markets(self) -> List[MarketCondition]:
        return self._markets


# ------------------------------------------------------------------
# CSV generation helper (creates sample data for testing)
# ------------------------------------------------------------------

def generate_sample_csv(
    path: str = "data/historical/btc_prices.csv",
    n_ticks: int = 86_400,  # 1 tick/sec = 1 day
) -> None:
    """Write a sample CSV file for backtesting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ticks = list(generate_synthetic_ticks(n_ticks=n_ticks))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "price"])
        writer.writeheader()
        for tick in ticks:
            writer.writerow({"timestamp_ms": tick.timestamp_ms, "price": tick.price})
    log.info("Wrote %d ticks to %s", len(ticks), path)
