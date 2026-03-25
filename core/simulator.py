"""
core/simulator.py — Live 3-minute simulation using real BTC prices and real
Polymarket markets.  No synthetic data, no real orders.

Run with:
    python main.py --mode simulate

What it does:
  1. Fetches real BTC price via Binance WebSocket
  2. Fetches real Polymarket BTC markets via Gamma API
  3. Runs the full signal pipeline for exactly 180 seconds (wall-clock)
  4. Simulates fills at market price ± slippage
  5. Evaluates WIN / LOSS / UNRESOLVED for each trade at end of window
  6. Prints a detailed trade-by-trade + aggregate summary
"""

import csv
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

from config import CONFIG
from core.condition_engine import ConditionEngine
from core.data_ingestion import (
    BinanceWebSocketFeed,
    PriceTick,
    PriceStore,
    bootstrap_price_store,
    PRICE_STORE,
)
from core.market_parser import MarketCondition, MarketParser
from core.pricing_engine import PricingEngine
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator, TradeSignal
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS

log = get_logger("simulator")

# How long to run the live simulation (seconds)
SIMULATION_DURATION_S: float = 180.0


# ------------------------------------------------------------------
# Trade record (sim only)
# ------------------------------------------------------------------

@dataclass
class SimTrade:
    signal_id: str
    question: str
    side: str                    # "YES" or "NO"
    entry_price: float           # Polymarket token price at fill
    entry_slippage: float        # Fraction applied
    btc_at_entry: float          # BTC price when signal fired
    size_usd: float
    expected_value_usd: float
    threshold_usd: Optional[float]
    condition_type: str
    # Filled in at settlement
    btc_at_exit: Optional[float] = None
    exit_price: Optional[float] = None
    result: str = "UNRESOLVED"   # "WIN" | "LOSS" | "UNRESOLVED"
    pnl_usd: float = 0.0

    def settle(self, final_btc_price: float) -> None:
        """
        Determine WIN/LOSS based on whether the real-world condition
        is met at the end of the simulation window.

        For "price_above X": WIN if BTC >= X (YES resolves 1.0)
        For "price_below X": WIN if BTC <= X (YES resolves 1.0)

        Exit price approximation:
          - WIN  → token resolves at 1.00, we bought at entry_price
          - LOSS → token resolves at 0.00
          - UNRESOLVED → we use current Polymarket mid (approximated as
            implied_certainty, which we don't have here, so we use
            entry_price ± small drift as a conservative estimate)
        """
        self.btc_at_exit = final_btc_price

        if self.threshold_usd is None:
            self.result = "UNRESOLVED"
            self.exit_price = self.entry_price
            self.pnl_usd = 0.0
            return

        # Determine if the YES condition is currently met
        if self.condition_type == "price_above":
            condition_met = final_btc_price >= self.threshold_usd
        elif self.condition_type == "price_below":
            condition_met = final_btc_price <= self.threshold_usd
        else:
            self.result = "UNRESOLVED"
            self.exit_price = self.entry_price
            self.pnl_usd = 0.0
            return

        # Apply to the side we traded
        if self.side == "YES":
            if condition_met:
                self.result = "WIN"
                self.exit_price = 1.0
            else:
                self.result = "LOSS"
                self.exit_price = 0.0
        else:  # NO token
            if not condition_met:
                self.result = "WIN"
                self.exit_price = 1.0
            else:
                self.result = "LOSS"
                self.exit_price = 0.0

        # PnL = (exit - entry_with_slippage) * size_usd
        # entry_price already includes slippage cost
        gross = (self.exit_price - self.entry_price) * self.size_usd
        fee = CONFIG.signal.taker_fee_fraction * self.size_usd
        self.pnl_usd = gross - fee


# ------------------------------------------------------------------
# Simulation result
# ------------------------------------------------------------------

@dataclass
class SimulationResult:
    total_ticks: int = 0
    total_signals: int = 0
    executed_trades: int = 0
    missed_opportunities: int = 0
    wins: int = 0
    losses: int = 0
    unresolved: int = 0
    total_pnl_usd: float = 0.0
    avg_ev_usd: float = 0.0
    runtime_s: float = 0.0
    trades: List[SimTrade] = field(default_factory=list)
    # Keep for backward compat with callers that expect .signals
    signals: List[dict] = field(default_factory=list)

    def print_summary(self) -> None:
        win_rate = (self.wins / self.executed_trades * 100) if self.executed_trades else 0.0

        print("\n" + "═" * 65)
        print("   📊  LIVE SIMULATION RESULTS  (3-minute window)")
        print("═" * 65)
        print(f"  Runtime:               {self.runtime_s:.1f}s")
        print(f"  BTC ticks processed:   {self.total_ticks:,}")
        print(f"  Signals generated:     {self.total_signals:,}")
        print(f"  Executed trades:       {self.executed_trades:,}")
        print(f"  Missed opportunities:  {self.missed_opportunities:,}")
        print(f"  ┌─ Winning trades:     {self.wins}")
        print(f"  ├─ Losing trades:      {self.losses}")
        print(f"  └─ Unresolved:         {self.unresolved}")
        print(f"  Win rate:              {win_rate:.1f}%")
        print(f"  Total simulated PnL:   ${self.total_pnl_usd:+.4f}")
        print(f"  Avg EV per trade:      ${self.avg_ev_usd:.4f}")
        print()

        if self.trades:
            print("  TRADE LOG")
            print("  " + "─" * 63)
            for t in self.trades:
                result_icon = {"WIN": "✅", "LOSS": "❌", "UNRESOLVED": "⏳"}.get(t.result, "?")
                threshold_str = f"${t.threshold_usd:,.0f}" if t.threshold_usd else "N/A"
                print(
                    f"  {result_icon} [{t.side:3s}] {t.question[:48]:<48}\n"
                    f"       threshold={threshold_str}  BTC@entry=${t.btc_at_entry:,.0f}"
                    f"  BTC@exit=${t.btc_at_exit:,.0f}\n"
                    f"       entry={t.entry_price:.3f}  exit={t.exit_price or 0:.3f}"
                    f"  size=${t.size_usd:.0f}  PnL=${t.pnl_usd:+.2f}"
                )
                print()
        else:
            print("  No trades executed during this window.")
            print("  (Edge thresholds may be too strict for current market)")

        print("═" * 65 + "\n")


# ------------------------------------------------------------------
# Live simulator
# ------------------------------------------------------------------

class Simulator:
    """
    Runs the full signal pipeline against live BTC prices and real
    Polymarket markets for SIMULATION_DURATION_S seconds.
    No real orders are placed.
    """

    def __init__(
        self,
        markets: Optional[List[MarketCondition]] = None,
        duration_s: float = SIMULATION_DURATION_S,
    ) -> None:
        self._forced_markets = markets  # if None → fetch from Polymarket
        self._duration_s = duration_s
        self._sim_trades: List[SimTrade] = []
        self._lock = threading.Lock()

    def add_market(self, market: MarketCondition) -> None:
        if self._forced_markets is None:
            self._forced_markets = []
        self._forced_markets.append(market)

    # ------------------------------------------------------------------

    def run(
        self,
        tick_source: Optional[Generator[PriceTick, None, None]] = None,
        max_ticks: Optional[int] = None,
        playback_delay_s: float = 0.0,
    ) -> SimulationResult:
        """
        Entry point.  tick_source / max_ticks / playback_delay_s are kept
        for backward-compatibility but ignored when running live.
        The live path is always used when tick_source is None (the default
        when called from main.py --mode simulate).
        """
        if tick_source is not None:
            # Legacy path: replay a generator (used by tests)
            return self._run_generator(tick_source, max_ticks, playback_delay_s)

        return self._run_live()

    # ------------------------------------------------------------------
    # Live path
    # ------------------------------------------------------------------

    def _run_live(self) -> SimulationResult:
        result = SimulationResult()
        start_time = time.time()
        deadline = start_time + self._duration_s

        # ── 1. Bootstrap price store via REST before WS connects ──────
        log.info("Bootstrapping BTC price via REST…")
        bootstrap_price_store()
        initial_tick = PRICE_STORE.latest()
        if initial_tick:
            log.info("Initial BTC price: $%.2f", initial_tick.price)
        else:
            log.warning("Could not bootstrap price — proceeding anyway")

        # ── 2. Start WebSocket feed ────────────────────────────────────
        feed = BinanceWebSocketFeed(store=PRICE_STORE)
        feed.start()
        log.info("WebSocket feed started. Waiting up to 5s for first live tick…")
        _wait_for_price(PRICE_STORE, timeout_s=5.0)

        # ── 3. Load Polymarket markets ─────────────────────────────────
        if self._forced_markets is not None:
            log.info("Using %d pre-loaded markets", len(self._forced_markets))
            parser = _StaticMarketParser(self._forced_markets)
        else:
            log.info("Fetching real Polymarket BTC markets…")
            parser = MarketParser()
            markets = parser.refresh()
            log.info("Loaded %d tradeable BTC markets from Polymarket", len(markets))
            if not markets:
                log.warning(
                    "No tradeable markets found — API may be rate-limiting or "
                    "all markets lack sufficient liquidity/time-to-expiry. "
                    "Proceeding with empty market set."
                )

        # ── 4. Build pipeline ─────────────────────────────────────────
        generator = SignalGenerator(
            market_parser=parser,
            capital_available_usd=CONFIG.risk.total_capital_usd,
        )
        # Shorten dedup window so we can re-evaluate during the 3-min run
        generator._dedup_window_s = 30.0

        risk = RiskManager()

        # ── 5. Main loop: process ticks until deadline ─────────────────
        log.info(
            "▶  Live simulation running for %.0f seconds…", self._duration_s
        )
        last_price = 0.0
        last_status_print = start_time

        while time.time() < deadline:
            tick = PRICE_STORE.latest()
            if tick is None or tick.is_stale:
                time.sleep(0.05)
                continue

            # Only process if price actually changed (avoid CPU spin)
            if tick.price == last_price:
                time.sleep(0.05)
                continue

            last_price = tick.price
            result.total_ticks += 1

            signals = generator.run(tick)
            result.total_signals += len(signals)

            for sig in signals:
                self._execute_sim_trade(sig, tick, risk, result)

            # Print a live status line every 15 seconds
            if time.time() - last_status_print >= 15.0:
                remaining = max(0, deadline - time.time())
                log.info(
                    "⏱  %.0fs remaining | BTC=$%.2f | ticks=%d signals=%d trades=%d",
                    remaining,
                    tick.price,
                    result.total_ticks,
                    result.total_signals,
                    result.executed_trades,
                )
                last_status_print = time.time()

        # ── 6. Stop feed ───────────────────────────────────────────────
        feed.stop()
        log.info("Feed stopped. Settling trades…")

        # ── 7. Settle all trades against final BTC price ───────────────
        final_tick = PRICE_STORE.latest()
        final_btc = final_tick.price if final_tick else 0.0
        log.info("Final BTC price for settlement: $%.2f", final_btc)

        for trade in self._sim_trades:
            trade.settle(final_btc)
            result.signals.append({
                "signal_id": trade.signal_id,
                "implied_certainty": 0.9,  # proxy for backward compat
                "net_edge": 0.0,
            })

        # ── 8. Compute aggregate stats ─────────────────────────────────
        ledger_summary = LEDGER.summary()
        result.missed_opportunities = max(
            0, ledger_summary["total_signals"] - result.executed_trades
        )
        result.trades = self._sim_trades
        result.wins = sum(1 for t in self._sim_trades if t.result == "WIN")
        result.losses = sum(1 for t in self._sim_trades if t.result == "LOSS")
        result.unresolved = sum(1 for t in self._sim_trades if t.result == "UNRESOLVED")
        result.total_pnl_usd = sum(t.pnl_usd for t in self._sim_trades)
        result.avg_ev_usd = (
            sum(t.expected_value_usd for t in self._sim_trades) / len(self._sim_trades)
            if self._sim_trades else 0.0
        )
        result.runtime_s = time.time() - start_time

        return result

    # ------------------------------------------------------------------
    # Simulated fill
    # ------------------------------------------------------------------

    def _execute_sim_trade(
        self,
        signal: TradeSignal,
        tick: PriceTick,
        risk: RiskManager,
        result: SimulationResult,
    ) -> None:
        decision = risk.approve(signal)
        if not decision.approved:
            return

        size_usd = decision.capped_size_usd

        # Slippage: base market-impact + random noise, capped at max_slippage
        slippage = min(
            CONFIG.execution.max_slippage_fraction,
            abs(random.gauss(0.005, 0.002)),  # mean 0.5%, std 0.2%
        )
        fill_price = signal.market_price * (1 + slippage)

        # Find the market condition to get threshold + condition_type
        market_condition = _resolve_market_condition(signal, risk)

        trade = SimTrade(
            signal_id=signal.signal_id,
            question=signal.question,
            side=signal.target_outcome,
            entry_price=fill_price,
            entry_slippage=slippage,
            btc_at_entry=tick.price,
            size_usd=size_usd,
            expected_value_usd=signal.expected_value_usd,
            threshold_usd=market_condition.get("threshold_usd"),
            condition_type=market_condition.get("condition_type", "unknown"),
        )

        with self._lock:
            self._sim_trades.append(trade)

        result.executed_trades += 1

        # Register with risk manager so position limits are respected
        from core.risk_manager import OpenPosition
        risk.register_open_position(OpenPosition(
            signal_id=signal.signal_id,
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.target_outcome,
            size_usd=size_usd,
            entry_price=fill_price,
        ))

        log.info(
            "💰 SIM FILL  [%s] %s | side=%s fill=%.3f slippage=%.2f%% size=$%.0f",
            signal.signal_id,
            signal.question[:50],
            signal.target_outcome,
            fill_price,
            slippage * 100,
            size_usd,
        )

    # ------------------------------------------------------------------
    # Legacy generator path (for tests / CSV replay)
    # ------------------------------------------------------------------

    def _run_generator(
        self,
        tick_source: Generator[PriceTick, None, None],
        max_ticks: Optional[int],
        playback_delay_s: float,
    ) -> SimulationResult:
        """Backward-compatible tick-based path used by unit tests."""
        parser = _StaticMarketParser(self._forced_markets or [])
        gen = SignalGenerator(
            market_parser=parser,
            capital_available_usd=CONFIG.risk.total_capital_usd,
        )
        risk = RiskManager()
        result = SimulationResult()
        start_time = time.time()

        for tick in tick_source:
            if max_ticks and result.total_ticks >= max_ticks:
                break
            result.total_ticks += 1
            if playback_delay_s > 0:
                time.sleep(playback_delay_s)
            signals = gen.run(tick)
            result.total_signals += len(signals)
            for sig in signals:
                result.signals.append(sig.to_dict())

        result.runtime_s = time.time() - start_time
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _wait_for_price(store: PriceStore, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if store.latest() is not None:
            return True
        time.sleep(0.1)
    return False


def _resolve_market_condition(signal: TradeSignal, risk: RiskManager) -> dict:
    """
    Extract threshold + condition_type from the signal's condition_notes
    (already embedded as a string by ConditionEngine).  Falls back to
    parsing the question directly.
    """
    from core.market_parser import _parse_threshold
    threshold, ctype = _parse_threshold(signal.question)
    return {"threshold_usd": threshold, "condition_type": ctype}


class _StaticMarketParser:
    """Wraps a fixed list of MarketCondition objects (used in tests and legacy path)."""

    def __init__(self, markets: List[MarketCondition]) -> None:
        self._markets = markets

    def needs_refresh(self) -> bool:
        return False

    def refresh(self) -> List[MarketCondition]:
        return self._markets

    def get_markets(self) -> List[MarketCondition]:
        return self._markets


# ------------------------------------------------------------------
# CSV helpers (kept for backward compat / generate-data mode)
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
    """Generate synthetic BTC prices via Geometric Brownian Motion (kept for tests)."""
    dt = tick_interval_ms / 1000 / (365 * 24 * 3600)
    sigma = annual_vol
    mu = drift
    price = start_price
    ts = int(time.time() * 1000)
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


def generate_sample_csv(
    path: str = "data/historical/btc_prices.csv",
    n_ticks: int = 86_400,
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
