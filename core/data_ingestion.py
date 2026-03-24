"""
core/data_ingestion.py — Real-time BTC price feed via Binance WebSocket.

Design notes:
- Runs in a background thread; main loop reads from a thread-safe price store.
- Auto-reconnects on disconnect with exponential backoff.
- Tracks last-update timestamp so callers can detect stale data.
- Also supports a REST fallback for bootstrapping and simulation mode.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import websocket  # websocket-client

from config import CONFIG
from utils.logger import get_logger
from utils.metrics import METRICS

log = get_logger("data_ingestion")


@dataclass
class PriceTick:
    symbol: str
    price: float
    timestamp_ms: int  # Exchange-reported trade time
    received_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def age_ms(self) -> int:
        return int(time.time() * 1000) - self.received_at_ms

    @property
    def is_stale(self) -> bool:
        threshold_ms = CONFIG.data_feed.price_staleness_threshold_s * 1000
        return self.age_ms > threshold_ms


class PriceStore:
    """
    Thread-safe singleton that holds the latest price tick.
    Multiple consumers can call `latest()` concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tick: Optional[PriceTick] = None
        self._callbacks: list[Callable[[PriceTick], None]] = []

    def update(self, tick: PriceTick) -> None:
        with self._lock:
            self._tick = tick
        METRICS.gauge("btc_price_usd", tick.price)
        METRICS.gauge("btc_price_age_ms", tick.age_ms)
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception as exc:
                log.error("Price callback error: %s", exc)

    def latest(self) -> Optional[PriceTick]:
        with self._lock:
            return self._tick

    def register_callback(self, fn: Callable[[PriceTick], None]) -> None:
        """Register a function called on every new tick (in ingestion thread)."""
        self._callbacks.append(fn)


# Global store — imported by other modules
PRICE_STORE = PriceStore()


class BinanceWebSocketFeed:
    """
    Connects to Binance trade stream and feeds PriceTick into PRICE_STORE.
    Runs in a daemon thread — lifecycle managed by the caller.
    """

    def __init__(self, store: PriceStore = PRICE_STORE) -> None:
        self._store = store
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._attempt = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-feed")
        self._thread.start()
        log.info("BinanceWebSocketFeed started")

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()
        log.info("BinanceWebSocketFeed stopped")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        cfg = CONFIG.data_feed
        while self._running:
            try:
                self._attempt += 1
                log.info("WS connect attempt #%d", self._attempt)
                self._ws = websocket.WebSocketApp(
                    cfg.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                log.error("WS exception: %s", exc)

            if not self._running:
                break

            delay = min(cfg.reconnect_delay_s * (2 ** min(self._attempt - 1, 6)), 60)
            if self._attempt >= cfg.max_reconnect_attempts:
                log.critical("Max WS reconnect attempts reached. Stopping feed.")
                self._running = False
                break
            log.warning("WS disconnected. Reconnecting in %.1fs…", delay)
            METRICS.inc("ws_reconnects")
            time.sleep(delay)

    def _on_open(self, ws) -> None:
        self._attempt = 0  # reset backoff on success
        log.info("WS connection established")
        METRICS.inc("ws_connects")

    def _on_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
            # Binance trade stream format: {"p": "price", "T": trade_time_ms, "s": symbol}
            tick = PriceTick(
                symbol=msg.get("s", "BTCUSDT"),
                price=float(msg["p"]),
                timestamp_ms=int(msg["T"]),
            )
            self._store.update(tick)
            METRICS.inc("price_ticks_received")
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Malformed WS message: %s | raw=%s", exc, raw[:200])
            METRICS.inc("price_ticks_malformed")

    def _on_error(self, ws, error) -> None:
        log.error("WS error: %s", error)
        METRICS.inc("ws_errors")

    def _on_close(self, ws, code, msg) -> None:
        log.warning("WS closed: code=%s msg=%s", code, msg)


# ------------------------------------------------------------------
# REST fallback (bootstrap / simulation)
# ------------------------------------------------------------------

def fetch_btc_price_rest() -> Optional[float]:
    """
    One-shot REST call to Binance for the current BTC price.
    Used to bootstrap the price store before the WS connects,
    and as a fallback during reconnect windows.
    """
    import urllib.request
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return float(data["price"])
    except Exception as exc:
        log.error("REST price fetch failed: %s", exc)
        return None


def bootstrap_price_store() -> None:
    """Seed the price store synchronously before the WS feed connects."""
    price = fetch_btc_price_rest()
    if price:
        tick = PriceTick(symbol="BTCUSDT", price=price, timestamp_ms=int(time.time() * 1000))
        PRICE_STORE.update(tick)
        log.info("Bootstrapped BTC price: $%.2f", price)
    else:
        log.warning("Could not bootstrap price store via REST")
