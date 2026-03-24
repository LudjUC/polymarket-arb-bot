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
    generate_synthetic_ticks,
    ticks_from_csv,
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
# Simulation mode
# ------------------------------------------------------------------

def run_simulation(
    data_path: Optional[str] = None,
    max_ticks: Optional[int] = None,
) -> None:
    log.info("=" * 60)
    log.info("  Polymarket Arb Bot — SIMULATION MODE")
    log.info("=" * 60)

    # Create some test markets with BTC thresholds near typical prices
    # In production, load these from a real Polymarket snapshot
    from core.market_parser import MarketCondition
    import time as _time

    test_markets = [
        MarketCondition(
            market_id="test_market_80k",
            question="Will Bitcoin reach $80,000 before end of Q1 2025?",
            condition_type="price_above",
            threshold_usd=80_000.0,
            expiry_ts=int(_time.time()) + 86400 * 30,
            yes_price=0.62,  # Polymarket prices YES at 62 cents
            no_price=0.38,
            yes_token_id="token_yes_80k",
            no_token_id="token_no_80k",
            liquidity_usd=25_000.0,
            raw={},
        ),
        MarketCondition(
            market_id="test_market_100k",
            question="Will BTC exceed $100,000 in 2025?",
            condition_type="price_above",
            threshold_usd=100_000.0,
            expiry_ts=int(_time.time()) + 86400 * 90,
            yes_price=0.45,
            no_price=0.55,
            yes_token_id="token_yes_100k",
            no_token_id="token_no_100k",
            liquidity_usd=50_000.0,
            raw={},
        ),
        MarketCondition(
            market_id="test_market_70k_below",
            question="Will Bitcoin fall below $70,000 before June 2025?",
            condition_type="price_below",
            threshold_usd=70_000.0,
            expiry_ts=int(_time.time()) + 86400 * 60,
            yes_price=0.20,
            no_price=0.80,
            yes_token_id="token_yes_70k_low",
            no_token_id="token_no_70k_low",
            liquidity_usd=15_000.0,
            raw={},
        ),
    ]

    sim = Simulator(markets=test_markets)

    if data_path:
        log.info("Loading historical data from %s", data_path)
        tick_source = ticks_from_csv(data_path)
    else:
        log.info("No data path given — using synthetic GBM ticks (start=$85,000)")
        tick_source = generate_synthetic_ticks(
            start_price=85_000.0,
            n_ticks=max_ticks or 5_000,
        )

    result = sim.run(
        tick_source=tick_source,
        max_ticks=max_ticks,
        playback_delay_s=0.0,  # run as fast as possible
    )

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
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  ws_reconnects: {metrics_snap['counters'].get('ws_reconnects', 0)}")
    print(f"  api_errors: {metrics_snap['counters'].get('api_errors', 0)}")
    print("=" * 60 + "\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "simulate", "generate-data"],
        default="simulate",
        help="Operating mode",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to historical BTC price CSV (simulate mode)",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Print signals only, do not place orders",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum ticks to process (simulation mode)",
    )
    args = parser.parse_args()

    if args.mode == "generate-data":
        generate_sample_csv()
        log.info("Sample data generated. Re-run with --mode simulate --data data/historical/btc_prices.csv")
        sys.exit(0)

    elif args.mode == "simulate":
        run_simulation(data_path=args.data, max_ticks=args.max_ticks)

    elif args.mode == "live":
        # Force simulation_mode=True unless user explicitly opts out via config
        run_live(execute=not args.no_execute)


if __name__ == "__main__":
    main()
