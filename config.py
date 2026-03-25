"""
Central configuration for the Polymarket Arbitrage Bot.
Edit these values before running. Never commit API keys.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataFeedConfig:
    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    reconnect_delay_s: float = 2.0
    max_reconnect_attempts: int = 10
    price_staleness_threshold_s: float = 5.0


@dataclass
class PolymarketConfig:
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    private_key: Optional[str] = field(
        default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY")
    )
    chain_id: int = 137

    # ── Market filtering ──────────────────────────────────────────────
    # How often to refresh market list (seconds)
    market_refresh_interval_s: float = 60.0

    # Only trade markets expiring within this window.
    # SHORT-TERM ONLY: we want markets that resolve TODAY or THIS WEEK
    # so that Polymarket prices actually react to live BTC data.
    max_time_to_expiry_s: float = 72 * 3600   # 72 hours max
    min_time_to_expiry_s: float = 10 * 60     # 10 minutes min (not about to snap shut)

    # Minimum liquidity — keeps us out of ghost markets
    min_market_liquidity_usd: float = 500.0


@dataclass
class SignalConfig:
    # ── Mispricing thresholds ────────────────────────────────────────
    # BUY YES when:  BTC >= threshold  AND  yes_price < buy_yes_max_price
    # BUY NO  when:  BTC <  threshold  AND  yes_price > buy_no_min_yes_price
    buy_yes_max_price: float = 0.70    # market pricing YES at ≤70% when we think it's ~90%
    buy_no_min_yes_price: float = 0.30 # market pricing YES at ≥30% when we think it's ~5%

    # Minimum gap between our implied prob and market price to act
    min_edge: float = 0.10             # at least 10 cents of edge after fees

    # Polymarket taker fee ~2%
    taker_fee_fraction: float = 0.02

    # ── Momentum / fast-move boost ────────────────────────────────────
    # If BTC moved this much in the last N seconds, boost signal confidence
    momentum_pct_threshold: float = 0.005   # 0.5% move
    momentum_window_s: float = 120.0        # over 2 minutes

    # ── Implied certainty floor (replaces old min_implied_certainty) ──
    # For BUY YES: BTC must be at least this far above threshold (as fraction)
    min_distance_pct: float = 0.005   # 0.5% above/below threshold


@dataclass
class ExitConfig:
    """Controls the realistic Polymarket price drift simulation at exit."""
    # Target YES price after full market correction
    # (market won't reach 1.0 in minutes — this is realistic drift)
    yes_drift_target_min: float = 0.80   # conservative
    yes_drift_target_max: float = 0.92   # optimistic (used when BTC far from threshold)

    # How long we hold before exiting (seconds)
    min_hold_s: float = 60.0    # at least 1 minute
    max_hold_s: float = 300.0   # at most 5 minutes

    # Speed of Polymarket price correction.
    # Fraction of the gap closed per minute (e.g. 0.30 = 30% of gap per minute)
    # Empirically Polymarket adjusts slowly — humans have to notice and trade
    correction_speed_per_min: float = 0.25


@dataclass
class RiskConfig:
    max_capital_per_trade_usd: float = 200.0
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 4
    kelly_fraction: float = 0.20       # conservative fractional Kelly
    total_capital_usd: float = 5_000.0


@dataclass
class ExecutionConfig:
    simulation_mode: bool = True
    # Slippage range for sim fills
    slippage_min: float = 0.003   # 0.3%
    slippage_max: float = 0.010   # 1.0%
    order_timeout_s: float = 10.0
    max_fill_retries: int = 2


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"
    rotate_when: str = "midnight"
    backup_count: int = 30
    json_logs: bool = False


@dataclass
class SimulationConfig:
    historical_data_path: str = "data/historical/btc_prices.csv"
    simulated_latency_ms: float = 150.0


@dataclass
class BotConfig:
    data_feed: DataFeedConfig = field(default_factory=DataFeedConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    heartbeat_interval_s: float = 15.0


CONFIG = BotConfig()
