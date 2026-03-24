"""
Central configuration for the Polymarket Arbitrage Bot.
Edit these values before running. Never commit API keys.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataFeedConfig:
    # Binance WebSocket for BTC/USDT — no API key needed for public streams
    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    reconnect_delay_s: float = 2.0
    max_reconnect_attempts: int = 10
    price_staleness_threshold_s: float = 5.0  # Alert if no price update in N sec


@dataclass
class PolymarketConfig:
    # Public CLOB API — no key needed for reads
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    # For order placement (optional)
    private_key: Optional[str] = field(
        default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY")
    )
    chain_id: int = 137  # Polygon mainnet
    # How often to refresh market list (seconds)
    market_refresh_interval_s: float = 30.0
    # Only consider markets with at least this much open interest (USD)
    min_market_liquidity_usd: float = 5_000.0
    # Ignore markets expiring within N seconds (too close to resolve)
    min_time_to_expiry_s: float = 3600.0


@dataclass
class SignalConfig:
    # Minimum edge after fees to generate a signal
    # Polymarket taker fee ~ 2% of notional
    taker_fee_fraction: float = 0.02
    # Safety margin on top of fees (slippage, model error, etc.)
    safety_margin: float = 0.03
    # Minimum raw edge = fees + safety_margin = 0.05 by default
    @property
    def min_edge(self) -> float:
        return self.taker_fee_fraction + self.safety_margin

    # Only trade when implied certainty (based on real-world data) > this
    min_implied_certainty: float = 0.90


@dataclass
class RiskConfig:
    max_capital_per_trade_usd: float = 500.0
    max_daily_loss_usd: float = 1_000.0
    max_open_positions: int = 5
    # Fraction of edge used for Kelly sizing (fractional Kelly = safer)
    kelly_fraction: float = 0.25
    total_capital_usd: float = 10_000.0


@dataclass
class ExecutionConfig:
    # Simulation mode: no real orders placed
    simulation_mode: bool = True
    # Max slippage before aborting order (fraction of price)
    max_slippage_fraction: float = 0.01
    # Order timeout in seconds
    order_timeout_s: float = 10.0
    # Retry failed fills up to N times
    max_fill_retries: int = 2


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"
    # Rotate log files daily
    rotate_when: str = "midnight"
    backup_count: int = 30
    # Structured JSON logs for machine parsing
    json_logs: bool = True


@dataclass
class SimulationConfig:
    # Path to historical BTC price CSV (timestamp_ms, price)
    historical_data_path: str = "data/historical/btc_prices.csv"
    # Playback speed multiplier (1.0 = real-time, 0 = as fast as possible)
    playback_speed: float = 0.0
    # Simulated latency added to each signal (ms)
    simulated_latency_ms: float = 150.0


@dataclass
class BotConfig:
    data_feed: DataFeedConfig = field(default_factory=DataFeedConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Heartbeat interval for monitoring loop (seconds)
    heartbeat_interval_s: float = 10.0


# Singleton
CONFIG = BotConfig()
