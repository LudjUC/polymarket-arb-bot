"""
Central configuration for the Polymarket Arbitrage Bot.
Edit these values before running. Never commit API keys.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


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

    # ── Market refresh ────────────────────────────────────────────────
    market_refresh_interval_s: float = 60.0

    # ── Expiry window — relaxed to 14 days ───────────────────────────
    # Shorter markets (< 72h) get lower min_edge.
    # Longer markets (7+ days) require higher edge to compensate for reversal risk.
    max_time_to_expiry_s: float = 14 * 24 * 3600   # 14 days
    min_time_to_expiry_s: float = 10 * 60           # 10 minutes

    # Minimum liquidity
    min_market_liquidity_usd: float = 500.0

    # ── Asset scope ───────────────────────────────────────────────────
    # Primary: BTC. Fallback: ETH when too few BTC markets found.
    primary_keywords: List[str] = field(
        default_factory=lambda: ["bitcoin", "btc"]
    )
    fallback_keywords: List[str] = field(
        default_factory=lambda: ["ethereum", "eth"]
    )
    allow_eth_fallback: bool = True
    # Activate ETH fallback when live BTC market count drops below this
    eth_fallback_min_btc_markets: int = 2

    # ── Time-tier boundaries (seconds) ───────────────────────────────
    tier_short_s: float = 72 * 3600       # < 72h  → short
    tier_medium_s: float = 7 * 24 * 3600  # 72h–7d → medium  (> 7d → long)


@dataclass
class SignalConfig:
    # ── Price gates ───────────────────────────────────────────────────
    # BUY YES when: condition met  AND  yes_price < buy_yes_max_price
    # BUY NO  when: condition not met  AND  yes_price > buy_no_min_yes_price
    buy_yes_max_price: float = 0.70
    buy_no_min_yes_price: float = 0.30

    # ── Tiered minimum net edge ───────────────────────────────────────
    # Short < 72h: high certainty + fast resolution → 10% edge enough
    min_edge_short: float = 0.10
    # Medium 72h–7d: more time for reversal → need 18%
    min_edge_medium: float = 0.18
    # Long > 7d: high uncertainty → need 28%
    min_edge_long: float = 0.28

    # Backward-compat alias used by legacy tests
    @property
    def min_edge(self) -> float:
        return self.min_edge_short

    # Polymarket taker fee
    taker_fee_fraction: float = 0.02

    # ── Momentum ──────────────────────────────────────────────────────
    momentum_pct_threshold: float = 0.005    # 0.5% move in window triggers boost
    momentum_window_s: float = 120.0         # 2-minute rolling window
    momentum_signal_window_s: float = 300.0  # 5-minute window for opportunistic logger

    # Minimum BTC distance from threshold to trade (avoids boundary noise)
    min_distance_pct: float = 0.005   # 0.5%


@dataclass
class ExitConfig:
    yes_drift_target_min: float = 0.80
    yes_drift_target_max: float = 0.92
    min_hold_s: float = 60.0
    max_hold_s: float = 300.0
    correction_speed_per_min: float = 0.25


@dataclass
class RiskConfig:
    max_capital_per_trade_usd: float = 200.0
    max_daily_loss_usd: float = 500.0
    max_open_positions: int = 4
    kelly_fraction: float = 0.20
    total_capital_usd: float = 5_000.0


@dataclass
class ExecutionConfig:
    simulation_mode: bool = True
    slippage_min: float = 0.003
    slippage_max: float = 0.010
    order_timeout_s: float = 10.0
    max_fill_retries: int = 2
    max_slippage_fraction: float = 0.015  # for legacy execution.py compat


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
class MomentumConfig:
    """
    Opportunistic momentum logger — active when no Polymarket markets exist.
    Detects BTC moves that *would* be actionable and logs them, so you can
    validate strategy effectiveness even in a market-dry period.
    """
    min_move_pct: float = 0.010           # 1% move over window to log
    no_market_log_interval_s: float = 60.0  # rate-limit "waiting…" messages
    window_s: float = 300.0              # 5-minute measurement window


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
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    heartbeat_interval_s: float = 15.0


CONFIG = BotConfig()
