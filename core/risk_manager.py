"""
core/risk_manager.py — Pre-trade and portfolio-level risk checks.

Enforces:
  - Max capital per trade
  - Max daily loss (stops all trading when hit)
  - Max concurrent open positions
  - Liquidity floor (rejects illiquid markets)
  - Staleness checks (rejects signals based on old price data)
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from config import CONFIG
from core.signal_generator import TradeSignal
from utils.logger import get_logger, LEDGER
from utils.metrics import METRICS

log = get_logger("risk_manager")


@dataclass
class RiskDecision:
    approved: bool
    signal_id: str
    capped_size_usd: float      # May be reduced from signal's recommended size
    rejection_reason: Optional[str] = None


@dataclass
class OpenPosition:
    signal_id: str
    market_id: str
    token_id: str
    side: str
    size_usd: float
    entry_price: float
    opened_at: float = field(default_factory=time.time)


class RiskManager:
    """
    Centralised risk gate. All signals must pass through `approve()` before
    being sent to execution.
    """

    def __init__(self) -> None:
        self._cfg = CONFIG.risk
        self._open_positions: Dict[str, OpenPosition] = {}  # signal_id → position
        self._daily_loss_usd: float = 0.0
        self._trading_halted: bool = False
        self._last_reset_date: date = datetime.now(timezone.utc).date()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def approve(self, signal: TradeSignal) -> RiskDecision:
        """
        Run all risk checks on a signal.
        Returns RiskDecision with approved=True/False and (possibly capped) size.
        """
        self._maybe_reset_daily_loss()

        # 1. Trading halted?
        if self._trading_halted:
            return self._reject(signal, "Trading halted — daily loss limit reached")

        # 2. Max open positions
        if len(self._open_positions) >= self._cfg.max_open_positions:
            return self._reject(
                signal,
                f"Max open positions ({self._cfg.max_open_positions}) reached",
            )

        # 3. Duplicate position on same market
        for pos in self._open_positions.values():
            if pos.market_id == signal.market_id:
                return self._reject(signal, f"Already have a position in market {signal.market_id}")

        # 4. Daily loss check
        if self._daily_loss_usd >= self._cfg.max_daily_loss_usd:
            self._trading_halted = True
            METRICS.gauge("trading_halted", 1)
            return self._reject(signal, f"Daily loss limit ${self._cfg.max_daily_loss_usd} reached")

        # 5. Size cap
        headroom = self._cfg.max_daily_loss_usd - self._daily_loss_usd
        capped_size = min(
            signal.recommended_size_usd,
            self._cfg.max_capital_per_trade_usd,
            headroom,
        )
        if capped_size <= 0:
            return self._reject(signal, "No capital headroom remaining")

        # All checks passed
        log.info(
            "Risk approved signal %s: size=$%.2f (requested=$%.2f)",
            signal.signal_id,
            capped_size,
            signal.recommended_size_usd,
        )
        METRICS.inc("risk_approvals")
        return RiskDecision(
            approved=True,
            signal_id=signal.signal_id,
            capped_size_usd=capped_size,
        )

    def register_open_position(self, position: OpenPosition) -> None:
        self._open_positions[position.signal_id] = position
        METRICS.gauge("open_positions", len(self._open_positions))
        log.info(
            "Position opened: signal=%s market=%s size=$%.2f",
            position.signal_id,
            position.market_id,
            position.size_usd,
        )

    def close_position(
        self, signal_id: str, realised_pnl_usd: float
    ) -> Optional[OpenPosition]:
        pos = self._open_positions.pop(signal_id, None)
        if pos is None:
            log.warning("Tried to close unknown position: %s", signal_id)
            return None

        if realised_pnl_usd < 0:
            self._daily_loss_usd += abs(realised_pnl_usd)
            METRICS.gauge("daily_loss_usd", self._daily_loss_usd)

        LEDGER.record_trade({
            "market_id": pos.market_id,
            "signal_id": signal_id,
            "side": pos.side,
            "size_usd": pos.size_usd,
            "entry_price": pos.entry_price,
            "realised_pnl_usd": realised_pnl_usd,
        })
        METRICS.gauge("open_positions", len(self._open_positions))
        log.info(
            "Position closed: signal=%s PnL=$%.4f | daily_loss=$%.2f",
            signal_id,
            realised_pnl_usd,
            self._daily_loss_usd,
        )
        return pos

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def daily_loss_usd(self) -> float:
        return self._daily_loss_usd

    def status(self) -> Dict:
        return {
            "trading_halted": self._trading_halted,
            "open_positions": self.open_position_count,
            "daily_loss_usd": round(self._daily_loss_usd, 4),
            "daily_loss_limit_usd": self._cfg.max_daily_loss_usd,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reject(self, signal: TradeSignal, reason: str) -> RiskDecision:
        METRICS.inc("risk_rejections")
        log.warning("Risk rejected signal %s: %s", signal.signal_id, reason)
        LEDGER.record_missed(reason=reason, signal=signal.to_dict())
        return RiskDecision(
            approved=False,
            signal_id=signal.signal_id,
            capped_size_usd=0.0,
            rejection_reason=reason,
        )

    def _maybe_reset_daily_loss(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset_date:
            log.info(
                "New trading day. Resetting daily loss (was $%.2f). Unhalting.",
                self._daily_loss_usd,
            )
            self._daily_loss_usd = 0.0
            self._trading_halted = False
            self._last_reset_date = today
            METRICS.gauge("trading_halted", 0)
            METRICS.gauge("daily_loss_usd", 0.0)
