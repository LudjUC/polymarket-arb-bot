"""
main.py — Bot entry point.

Usage:
  # Live mode (reads real BTC prices, uses real Polymarket markets):
  python main.py --mode live

  # Simulation with synthetic data:
  python main.py --mode simulate

  # Simulation with historical CSV:
  python main.py --mode simulate --data data/historical/btc_prices.csv

  # Generate sample CSV for testing:
  python main.py --mode generate-data

  # Dry-run: connects to real feeds but prints signals without executing:
  python main.py --mode live --no-execute

Flags:
  --mode          live | simulate | generate-data
  --data          Path to historical CSV (simulate mode)
  --no-execute    Print signals only, no orders
  --max-ticks     Limit ticks in simulation mode
"""

import argparse
import signal
import sys
import time
from typing import Optional

from config import CONFIG
from core.data_ingestion import BinanceWebSocketFeed, bootstrap_price_store, PRICE_STORE
from core.execution import ExecutionEngine
from core.market_parser import MarketParser
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator, TradeSignal
from core.simulator import (
    Simulator,
    generate_sample_csv,
)
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS

log = get_logger("main")


# ------------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------------

_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    log.info("Shutdown signal received (%s). Draining…", sig)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ------------------------------------------------------------------
# Live mode
# ------------------------------------------------------------------

def run_live(execute: bool = True) -> None:
    log.info("=" * 60)
    log.info("  Polymarket Arb Bot — LIVE MODE (execute=%s)", execute)
    if CONFIG.execution.simulation_mode and execute:
        log.info("  NOTE: execution.simulation_mode=True — orders are simulated")
    log.info("=" * 60)

    # Bootstrap price before WS connects
    bootstrap_price_store()

    # Start WebSocket feed
    feed = BinanceWebSocketFeed()
    feed.start()

    # Build pipeline
    parser = MarketParser()
    risk = RiskManager()
    generator = SignalGenerator(
        market_parser=parser,
        capital_available_usd=CONFIG.risk.total_capital_usd,
    )
    executor = ExecutionEngine(risk_manager=risk)

    # Initial market load
    log.info("Loading Polymarket markets…")
    parser.refresh()

    # Register price callback: evaluate every tick
    def on_tick(tick):
        if _shutdown:
            return
        signals = generator.run(tick)
        if not execute:
            for sig in signals:
                _print_signal(sig)
            return

        for sig in signals:
            result = executor.execute(sig)
            if result:
                log.info(
                    "Order result: signal=%s status=%s fill_price=%.4f size=$%.2f",
                    result.signal_id,
                    result.status.name,
                    result.fill_price,
                    result.filled_size_usd,
                )

    PRICE_STORE.register_callback(on_tick)

    # Main loop: heartbeat + monitoring
    try:
        while not _shutdown:
            time.sleep(CONFIG.heartbeat_interval_s)
            _heartbeat(risk)
    finally:
        feed.stop()
        _print_final_summary()


# ------------------------------------------------------------------
# Simulation mode  (live time-based window with real data)
# ------------------------------------------------------------------

def run_simulation(duration_s: float = 180.0) -> None:
    """
    Run a live simulation for `duration_s` seconds (default 3 minutes).
      - Real BTC prices from Binance WebSocket
      - Real Polymarket BTC markets from Gamma API
      - Simulated fills with slippage (no real money)
      - WIN/LOSS settled against final BTC price at end of window
    """
    minutes = int(duration_s // 60)
    secs    = int(duration_s % 60)
    dur_str = f"{minutes}m {secs:02d}s" if secs else f"{minutes}m"

    print("\n" + "═" * 65)
    print("   🚀  POLYMARKET ARB BOT — LIVE SIMULATION MODE")
    print(f"   Duration : {dur_str}  |  Real data  |  No real money")
    print("   Feed     : Binance WebSocket (BTC/USDT)")
    print("   Markets  : Polymarket Gamma API (live)")
    print("═" * 65 + "\n")

    sim = Simulator(duration_s=duration_s)
    result = sim.run()   # always uses live WS path when tick_source=None
    result.print_summary()
    _print_final_summary()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _heartbeat(risk: RiskManager) -> None:
    tick = PRICE_STORE.latest()
    price_str = f"${tick.price:,.2f}" if tick else "N/A"
    age_str = f"{tick.age_ms}ms" if tick else "N/A"
    status = risk.status()
    log.info(
        "HEARTBEAT | BTC=%s (age=%s) | open_pos=%d | daily_loss=$%.2f | halted=%s | "
        "signals=%d fills=%d",
        price_str,
        age_str,
        status["open_positions"],
        status["daily_loss_usd"],
        status["trading_halted"],
        METRICS.get_counter("signals_generated"),
        METRICS.get_counter("orders_filled"),
    )
    snapshot = METRICS.snapshot()
    sig_lat = snapshot["histograms"].get("signal_generation_ms", {})
    if sig_lat:
        log.debug(
            "Signal latency p50=%.1fms p95=%.1fms p99=%.1fms",
            sig_lat.get("p50", 0),
            sig_lat.get("p95", 0),
            sig_lat.get("p99", 0),
        )


def _print_signal(sig: TradeSignal) -> None:
    print(
        f"\n{'─'*60}\n"
        f"  📊 SIGNAL  [{sig.signal_id}]\n"
        f"  Market:    {sig.question[:70]}\n"
        f"  Side:      {sig.target_outcome}\n"
        f"  BTC price: ${sig.btc_price_at_signal:,.2f}\n"
        f"  Mkt price: {sig.market_price:.3f}   Certainty: {sig.implied_certainty:.3f}\n"
        f"  Net edge:  {sig.net_edge:.3f}   Size: ${sig.recommended_size_usd:.2f}\n"
        f"  EV:        ${sig.expected_value_usd:.4f}\n"
        f"  Notes:     {sig.condition_notes}\n"
        f"{'─'*60}"
    )


def _print_final_summary() -> None:
    summary = LEDGER.summary()
    metrics_snap = METRICS.snapshot()
    print("─" * 65)
    print("  LEDGER TOTALS")
    print(f"  Signals recorded:  {summary['total_signals']}")
    print(f"  Trades recorded:   {summary['total_trades']}")
    print(f"  Realised PnL:      ${summary['total_realised_pnl_usd']:+.4f}")
    ws_reconnects = int(metrics_snap["counters"].get("ws_reconnects", 0))
    api_errors = int(metrics_snap["counters"].get("api_errors", 0))
    if ws_reconnects:
        print(f"  WS reconnects:     {ws_reconnects}")
    if api_errors:
        print(f"  API errors:        {api_errors}")
    print("─" * 65 + "\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket Arbitrage Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode simulate              # 3-min live sim\n"
            "  python main.py --mode simulate --duration 60  # 1-min live sim\n"
            "  python main.py --mode live --no-execute     # live signals, no orders\n"
            "  python main.py --mode generate-data         # write sample CSV\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["live", "simulate", "generate-data"],
        default="simulate",
        help="Operating mode (default: simulate)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=180.0,
        metavar="SECONDS",
        help="Simulation window in seconds (default: 180 = 3 min)",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="(live mode) Print signals only, do not place orders",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set log level to DEBUG (shows market rejection reasons)",
    )
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        for name in ("market_parser", "signal_generator", "condition_engine", "pricing_engine"):
            logging.getLogger(name).setLevel(logging.DEBUG)

    if args.mode == "generate-data":
        generate_sample_csv()
        log.info("Sample CSV written. Use --mode simulate to run live simulation.")
        sys.exit(0)

    elif args.mode == "simulate":
        run_simulation(duration_s=args.duration)

    elif args.mode == "live":
        run_live(execute=not args.no_execute)


if __name__ == "__main__":
    main()
