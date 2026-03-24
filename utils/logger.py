"""
utils/logger.py — Structured logging + PnL / opportunity tracking.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, Optional

from config import CONFIG


class JsonFormatter(logging.Formatter):
    """Emits each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via `extra=`
        for key, val in record.__dict__.items():
            if key.startswith("_"):
                payload[key[1:]] = val
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the bot."""
    cfg = CONFIG.logging
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(cfg.log_level)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(cfg.log_level)

    # File handler (rotating)
    fh = TimedRotatingFileHandler(
        filename=os.path.join(cfg.log_dir, "bot.log"),
        when=cfg.rotate_when,
        backupCount=cfg.backup_count,
        utc=True,
    )
    fh.setLevel(cfg.log_level)

    if cfg.json_logs:
        fmt = JsonFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# PnL & opportunity ledger
# ---------------------------------------------------------------------------

class Ledger:
    """
    In-memory ledger that tracks:
    - Signals generated
    - Trades executed (sim or real)
    - PnL realised/unrealised
    - Missed opportunities (signal generated but not filled)
    """

    def __init__(self) -> None:
        self._log = get_logger("ledger")
        self.signals: list[Dict] = []
        self.trades: list[Dict] = []
        self.daily_loss_usd: float = 0.0
        self._ledger_path = os.path.join(CONFIG.logging.log_dir, "ledger.jsonl")

    # ------------------------------------------------------------------
    def record_signal(self, signal: Dict) -> None:
        signal["recorded_at"] = datetime.now(timezone.utc).isoformat()
        self.signals.append(signal)
        self._append({"type": "signal", **signal})
        self._log.info(
            "Signal detected",
            extra={
                "_market_id": signal.get("market_id"),
                "_edge": signal.get("edge"),
                "_ev": signal.get("expected_value"),
            },
        )

    def record_trade(self, trade: Dict) -> None:
        trade["recorded_at"] = datetime.now(timezone.utc).isoformat()
        self.trades.append(trade)
        pnl = trade.get("realised_pnl_usd", 0.0)
        if pnl < 0:
            self.daily_loss_usd += abs(pnl)
        self._append({"type": "trade", **trade})
        self._log.info(
            "Trade recorded",
            extra={
                "_market_id": trade.get("market_id"),
                "_side": trade.get("side"),
                "_size_usd": trade.get("size_usd"),
                "_realised_pnl": pnl,
            },
        )

    def record_missed(self, reason: str, signal: Dict) -> None:
        entry = {
            "type": "missed",
            "reason": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **signal,
        }
        self._append(entry)
        self._log.warning("Missed opportunity: %s", reason, extra={"_market_id": signal.get("market_id")})

    def summary(self) -> Dict:
        total_pnl = sum(t.get("realised_pnl_usd", 0.0) for t in self.trades)
        return {
            "total_signals": len(self.signals),
            "total_trades": len(self.trades),
            "total_realised_pnl_usd": round(total_pnl, 4),
            "daily_loss_usd": round(self.daily_loss_usd, 4),
        }

    # ------------------------------------------------------------------
    def _append(self, record: Dict) -> None:
        try:
            with open(self._ledger_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            self._log.error("Failed to write ledger: %s", exc)


# Global ledger instance
LEDGER = Ledger()
