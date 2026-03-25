"""
core/simulator.py — Live simulation with realistic short-term exit pricing.

Strategy summary:
  1. Connect to Binance WS for live BTC price
  2. Load Polymarket BTC/ETH markets (expiring within configured window, up to 14 days)
  3. Detect mispricing: spot price already above/below threshold but Polymarket
     hasn't repriced yet
  4. Simulate entry + gradual exit (Polymarket price drifts toward fair value)
  5. Track full metrics: PnL, hold time, drawdown, win rate

Exit model:
  We do NOT resolve trades to 1.0/0.0 instantly.
  Instead we simulate Polymarket price gradually correcting over the hold period.
  Target exit price is based on how far spot is from the threshold at exit time.
  Correction follows exponential decay: ~25% of remaining gap closed per minute.
"""

import csv
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

from config import CONFIG
from core.condition_engine import ConditionEngine, PRICE_HISTORY
from core.data_ingestion import (
    BinanceWebSocketFeed, PriceTick, PriceStore,
    bootstrap_price_store, PRICE_STORE,
)
from core.market_parser import MarketCondition, MarketParser
from core.pricing_engine import PricingEngine
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator, TradeSignal
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS

log = get_logger("simulator")

SIMULATION_DURATION_S: float = 180.0


# ---------------------------------------------------------------------------
# SimTrade — with realistic drift-based exit
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    signal_id: str
    question: str
    side: str                      # "YES" or "NO"
    entry_price: float
    entry_slippage: float
    btc_at_entry: float
    size_usd: float
    expected_value_usd: float
    threshold_usd: Optional[float]
    condition_type: str
    implied_certainty_at_entry: float
    expiry_tier: str = "short"
    entered_at: float = field(default_factory=time.time)

    btc_at_exit: Optional[float] = None
    exit_price: Optional[float] = None
    exited_at: Optional[float] = None
    result: str = "UNRESOLVED"
    pnl_usd: float = 0.0
    hold_time_s: float = 0.0

    def compute_target_exit_price(self, current_btc: float) -> float:
        if self.threshold_usd is None or self.threshold_usd <= 0:
            return self.entry_price

        exit_cfg = CONFIG.exit

        if self.side == "YES":
            if self.condition_type == "price_above":
                dist = (current_btc - self.threshold_usd) / self.threshold_usd
            else:
                dist = (self.threshold_usd - current_btc) / self.threshold_usd
        else:
            if self.condition_type == "price_above":
                dist = (self.threshold_usd - current_btc) / self.threshold_usd
            else:
                dist = (current_btc - self.threshold_usd) / self.threshold_usd

        if dist <= 0:
            return max(0.05, self.entry_price - 0.15)

        target = exit_cfg.yes_drift_target_min + min(dist * 4.0, 0.12)
        target = min(target, exit_cfg.yes_drift_target_max)
        return round(target, 4)

    def simulate_exit(self, current_btc: float, elapsed_s: float) -> None:
        exit_cfg = CONFIG.exit
        fee = CONFIG.signal.taker_fee_fraction

        self.hold_time_s = elapsed_s
        self.btc_at_exit = current_btc
        self.exited_at = time.time()

        target = self.compute_target_exit_price(current_btc)

        k = exit_cfg.correction_speed_per_min / 60.0
        fraction_closed = 1.0 - math.exp(-k * elapsed_s)
        fraction_closed = min(fraction_closed, 0.80)

        self.exit_price = self.entry_price + (target - self.entry_price) * fraction_closed
        self.exit_price = max(0.01, min(0.99, round(self.exit_price, 4)))

        gross = (self.exit_price - self.entry_price) * self.size_usd
        total_fee = fee * self.size_usd
        self.pnl_usd = round(gross - total_fee, 4)

        if self.pnl_usd > 0:
            self.result = "WIN"
        elif self.pnl_usd < -total_fee * 0.5:
            self.result = "LOSS"
        else:
            self.result = "UNRESOLVED"


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

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
    avg_hold_time_s: float = 0.0
    max_drawdown_usd: float = 0.0
    runtime_s: float = 0.0
    trades: List[SimTrade] = field(default_factory=list)
    signals: List[dict] = field(default_factory=list)

    def print_summary(self) -> None:
        win_rate = (self.wins / self.executed_trades * 100) if self.executed_trades else 0.0
        avg_pnl  = (self.total_pnl_usd / self.executed_trades) if self.executed_trades else 0.0

        print("\n" + "═" * 65)
        print("   📊  SIMULATION RESULTS")
        print("═" * 65)
        print(f"  Runtime                {self.runtime_s:.1f}s")
        print(f"  BTC ticks processed    {self.total_ticks:,}")
        print(f"  Signals generated      {self.total_signals:,}")
        print(f"  Executed trades        {self.executed_trades:,}")
        print(f"  Missed (risk filtered) {self.missed_opportunities:,}")
        print(f"  ┌─ WIN                 {self.wins}")
        print(f"  ├─ LOSS               {self.losses}")
        print(f"  └─ UNRESOLVED         {self.unresolved}")
        print(f"  Win rate               {win_rate:.1f}%")
        print(f"  Total PnL              ${self.total_pnl_usd:+.4f}")
        print(f"  Avg profit/trade       ${avg_pnl:+.4f}")
        print(f"  Avg EV at entry        ${self.avg_ev_usd:.4f}")
        print(f"  Avg hold time          {self.avg_hold_time_s:.0f}s")
        print(f"  Max drawdown           ${self.max_drawdown_usd:.4f}")
        print()

        if self.trades:
            print("  TRADE LOG")
            print("  " + "─" * 63)
            for t in self.trades:
                icon = {"WIN": "✅", "LOSS": "❌", "UNRESOLVED": "⏳"}.get(t.result, "?")
                thr  = f"${t.threshold_usd:,.0f}" if t.threshold_usd else "N/A"
                tier_tag = f"[{t.expiry_tier}]" if t.expiry_tier else ""
                print(
                    f"  {icon} {tier_tag} [{t.side}]  {t.question[:48]}\n"
                    f"     threshold={thr}  BTC@entry=${t.btc_at_entry:,.0f}"
                    f"  BTC@exit=${t.btc_at_exit or 0:,.0f}\n"
                    f"     entry={t.entry_price:.3f}  exit={t.exit_price or 0:.3f}"
                    f"  size=${t.size_usd:.0f}  hold={t.hold_time_s:.0f}s"
                    f"  PnL=${t.pnl_usd:+.2f}"
                )
                if t.implied_certainty_at_entry:
                    print(f"     impl_prob={t.implied_certainty_at_entry:.3f}"
                          f"  entry_slippage={t.entry_slippage:.2%}")
                print()
        else:
            print("  No trades executed.")
            print()
            print("  Why no trades?")
            print(
                f"  • No Polymarket BTC/ETH markets passed the current filters.\n"
                f"    The bot searched for markets expiring within "
                f"{CONFIG.polymarket.max_time_to_expiry_s / 3600:.0f}h "
                f"with ≥${CONFIG.polymarket.min_market_liquidity_usd:.0f} liquidity."
            )
            print()
            print("  Possible fixes:")
            print("  • Run python debug_markets.py to inspect live market availability.")
            print("  • Polymarket may have no active BTC/ETH threshold markets today.")
            print("  • Check whether market question format has changed (new API schema).")
            print("  • Try running during high-volatility periods or near weekly")
            print("    resolution dates (Sunday/Monday for weekly markets).")

        print("═" * 65 + "\n")


# ---------------------------------------------------------------------------
# Live simulator
# ---------------------------------------------------------------------------

class Simulator:

    def __init__(
        self,
        markets: Optional[List[MarketCondition]] = None,
        duration_s: float = SIMULATION_DURATION_S,
    ) -> None:
        self._forced_markets = markets
        self._duration_s = duration_s
        self._sim_trades: List[SimTrade] = []
        self._lock = threading.Lock()

    def add_market(self, market: MarketCondition) -> None:
        if self._forced_markets is None:
            self._forced_markets = []
        self._forced_markets.append(market)

    def run(
        self,
        tick_source: Optional[Generator] = None,
        max_ticks: Optional[int] = None,
        playback_delay_s: float = 0.0,
    ) -> SimulationResult:
        if tick_source is not None:
            return self._run_generator(tick_source, max_ticks, playback_delay_s)
        return self._run_live()

    # -----------------------------------------------------------------------

    def _run_live(self) -> SimulationResult:
        result   = SimulationResult()
        start_ts = time.time()
        deadline = start_ts + self._duration_s

        log.info("Bootstrapping BTC price via REST…")
        bootstrap_price_store()
        initial = PRICE_STORE.latest()
        if initial:
            log.info("Initial BTC price: $%.2f", initial.price)

        feed = BinanceWebSocketFeed(store=PRICE_STORE)
        feed.start()
        log.info("WS feed started — waiting up to 5s for first live tick…")
        _wait_for_price(PRICE_STORE, 5.0)

        if self._forced_markets is not None:
            parser = _StaticMarketParser(self._forced_markets)
            log.info("Using %d pre-loaded markets", len(self._forced_markets))
        else:
            parser = MarketParser()
            parser.refresh()

        if not parser.get_markets():
            log.warning(
                "No tradeable markets found after initial refresh. "
                "The momentum logger will fire during the run to show "
                "hypothetical opportunities. Use python debug_markets.py "
                "to inspect live market availability."
            )

        gen = SignalGenerator(
            market_parser=parser,
            capital_available_usd=CONFIG.risk.total_capital_usd,
        )
        gen._dedup_window_s = 30.0
        risk = RiskManager()

        log.info("▶  Running for %.0fs…", self._duration_s)
        last_price     = 0.0
        last_status_ts = start_ts
        running_pnl    = 0.0

        while time.time() < deadline:
            tick = PRICE_STORE.latest()
            if tick is None or tick.is_stale:
                time.sleep(0.05)
                continue
            if tick.price == last_price:
                time.sleep(0.05)
                continue

            last_price = tick.price
            result.total_ticks += 1

            self._check_exits(tick, result)

            signals = gen.run(tick)
            result.total_signals += len(signals)
            for sig in signals:
                self._enter_trade(sig, tick, risk, result)

            if time.time() - last_status_ts >= 15.0:
                remaining = max(0, deadline - time.time())
                open_pos  = sum(1 for t in self._sim_trades if t.exited_at is None)
                log.info(
                    "⏱  %.0fs left | BTC=$%.2f | ticks=%d sigs=%d "
                    "trades=%d open=%d pnl=$%+.2f",
                    remaining, tick.price, result.total_ticks,
                    result.total_signals, result.executed_trades,
                    open_pos, running_pnl,
                )
                last_status_ts = time.time()

        final = PRICE_STORE.latest()
        final_btc = final.price if final else last_price
        log.info("Simulation ended. Settling open trades at BTC=$%.2f", final_btc)

        for trade in self._sim_trades:
            if trade.exited_at is None:
                elapsed = time.time() - trade.entered_at
                trade.simulate_exit(final_btc, elapsed)

        feed.stop()

        ledger_summary = LEDGER.summary()
        result.missed_opportunities = max(
            0, ledger_summary["total_signals"] - result.executed_trades
        )
        result.trades     = self._sim_trades
        result.wins       = sum(1 for t in self._sim_trades if t.result == "WIN")
        result.losses     = sum(1 for t in self._sim_trades if t.result == "LOSS")
        result.unresolved = sum(1 for t in self._sim_trades if t.result == "UNRESOLVED")
        result.total_pnl_usd = round(sum(t.pnl_usd for t in self._sim_trades), 4)

        if self._sim_trades:
            result.avg_ev_usd = sum(
                t.expected_value_usd for t in self._sim_trades
            ) / len(self._sim_trades)
            result.avg_hold_time_s = sum(
                t.hold_time_s for t in self._sim_trades
            ) / len(self._sim_trades)

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted(self._sim_trades, key=lambda x: x.exited_at or 0):
            cumulative += t.pnl_usd
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)
        result.max_drawdown_usd = round(max_dd, 4)
        result.runtime_s = time.time() - start_ts

        return result

    # -----------------------------------------------------------------------
    # Trade entry
    # -----------------------------------------------------------------------

    def _enter_trade(
        self, signal: TradeSignal, tick: PriceTick,
        risk: RiskManager, result: SimulationResult,
    ) -> None:
        decision = risk.approve(signal)
        if not decision.approved:
            return

        size_usd = decision.capped_size_usd

        slip_min = CONFIG.execution.slippage_min
        slip_max = CONFIG.execution.slippage_max
        slippage = random.uniform(slip_min, slip_max)
        fill_price = min(0.99, signal.market_price * (1 + slippage))

        from core.market_parser import _parse_threshold
        threshold, ctype = _parse_threshold(signal.question)

        trade = SimTrade(
            signal_id=signal.signal_id,
            question=signal.question,
            side=signal.target_outcome,
            entry_price=fill_price,
            entry_slippage=slippage,
            btc_at_entry=tick.price,
            size_usd=size_usd,
            expected_value_usd=signal.expected_value_usd,
            threshold_usd=threshold,
            condition_type=ctype or "price_above",
            implied_certainty_at_entry=signal.implied_certainty,
            expiry_tier=signal.expiry_tier,
        )

        with self._lock:
            self._sim_trades.append(trade)
        result.executed_trades += 1

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
            "💰 ENTER [%s][%s] %s | side=%s  entry=%.3f  slip=%.2f%%  "
            "size=$%.0f  impl=%.3f  BTC=$%.0f",
            signal.signal_id, signal.expiry_tier, signal.question[:35],
            signal.target_outcome, fill_price, slippage * 100,
            size_usd, signal.implied_certainty, tick.price,
        )

    # -----------------------------------------------------------------------
    # Exit checking
    # -----------------------------------------------------------------------

    def _check_exits(self, tick: PriceTick, result: SimulationResult) -> None:
        exit_cfg = CONFIG.exit
        now = time.time()

        with self._lock:
            open_trades = [t for t in self._sim_trades if t.exited_at is None]

        for trade in open_trades:
            elapsed = now - trade.entered_at
            if elapsed < exit_cfg.min_hold_s:
                continue

            should_exit = False
            exit_reason = ""

            if elapsed >= exit_cfg.max_hold_s:
                should_exit = True
                exit_reason = "max_hold_time"

            if not should_exit:
                target = trade.compute_target_exit_price(tick.price)
                current_sim = self._simulate_current_price(trade, tick.price, elapsed)
                if current_sim >= target * 0.95:
                    should_exit = True
                    exit_reason = "target_reached"

            if should_exit:
                trade.simulate_exit(tick.price, elapsed)
                log.info(
                    "🔚 EXIT  [%s] %s | %s  entry=%.3f→exit=%.3f  "
                    "hold=%.0fs  PnL=$%+.2f",
                    trade.signal_id, trade.question[:35], exit_reason,
                    trade.entry_price, trade.exit_price or 0,
                    elapsed, trade.pnl_usd,
                )

    def _simulate_current_price(
        self, trade: SimTrade, current_btc: float, elapsed_s: float
    ) -> float:
        k = CONFIG.exit.correction_speed_per_min / 60.0
        target = trade.compute_target_exit_price(current_btc)
        fraction = min(1.0 - math.exp(-k * elapsed_s), 0.80)
        return trade.entry_price + (target - trade.entry_price) * fraction

    # -----------------------------------------------------------------------
    # Legacy generator path (kept for unit tests)
    # -----------------------------------------------------------------------

    def _run_generator(
        self,
        tick_source: Generator,
        max_ticks: Optional[int],
        playback_delay_s: float,
    ) -> SimulationResult:
        parser = _StaticMarketParser(self._forced_markets or [])
        gen    = SignalGenerator(
            market_parser=parser,
            capital_available_usd=CONFIG.risk.total_capital_usd,
        )
        risk   = RiskManager()
        result = SimulationResult()
        start  = time.time()

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

        result.runtime_s = time.time() - start
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_price(store: PriceStore, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if store.latest() is not None:
            return True
        time.sleep(0.1)
    return False


class _StaticMarketParser:
    def __init__(self, markets: List[MarketCondition]) -> None:
        self._markets = markets
    def needs_refresh(self) -> bool:
        return False
    def refresh(self) -> List[MarketCondition]:
        return self._markets
    def get_markets(self) -> List[MarketCondition]:
        return self._markets


# ---------------------------------------------------------------------------
# CSV / synthetic helpers
# ---------------------------------------------------------------------------

def ticks_from_csv(path: str) -> Generator:
    if not os.path.exists(path):
        log.error("CSV not found: %s", path)
        return
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                yield PriceTick(
                    symbol="BTCUSDT",
                    price=float(row["price"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    received_at_ms=int(row["timestamp_ms"]),
                )
            except (KeyError, ValueError):
                pass


def generate_synthetic_ticks(
    start_price: float = 85_000.0,
    n_ticks: int = 10_000,
    tick_interval_ms: int = 1_000,
    annual_vol: float = 0.70,
    drift: float = 0.0,
) -> Generator:
    import math as _math
    dt    = tick_interval_ms / 1000 / (365 * 24 * 3600)
    price = start_price
    ts    = int(time.time() * 1000)
    for _ in range(n_ticks):
        z = random.gauss(0, 1)
        price *= _math.exp(
            (drift - 0.5 * annual_vol ** 2) * dt
            + annual_vol * _math.sqrt(dt) * z
        )
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ticks = list(generate_synthetic_ticks(n_ticks=n_ticks))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "price"])
        writer.writeheader()
        for t in ticks:
            writer.writerow({"timestamp_ms": t.timestamp_ms, "price": t.price})
    log.info("Wrote %d ticks to %s", len(ticks), path)
