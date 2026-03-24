"""
utils/metrics.py — Lightweight in-process metrics (counters, gauges, histograms).
No external dependency required; can be swapped for Prometheus if needed.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple


class Metrics:
    """Thread-safe, in-memory metrics store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        # histograms store last N values
        self._histograms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))

    # ------------------------------------------------------------------
    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Record a histogram observation (e.g., latency)."""
        with self._lock:
            self._histograms[name].append(value)

    # ------------------------------------------------------------------
    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> Optional[float]:
        with self._lock:
            return self._gauges.get(name)

    def percentile(self, name: str, pct: float) -> Optional[float]:
        """Return the p-th percentile of a histogram (0–100)."""
        with self._lock:
            data = sorted(self._histograms[name])
        if not data:
            return None
        idx = max(0, int(len(data) * pct / 100) - 1)
        return data[idx]

    def snapshot(self) -> Dict:
        with self._lock:
            hist_summary = {
                k: {
                    "count": len(v),
                    "p50": self._pct(v, 50),
                    "p95": self._pct(v, 95),
                    "p99": self._pct(v, 99),
                }
                for k, v in self._histograms.items()
            }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": hist_summary,
            }

    @staticmethod
    def _pct(data: Deque[float], pct: float) -> Optional[float]:
        s = sorted(data)
        if not s:
            return None
        idx = max(0, int(len(s) * pct / 100) - 1)
        return round(s[idx], 6)


# Global metrics instance
METRICS = Metrics()


# ------------------------------------------------------------------
# Latency context manager
# ------------------------------------------------------------------

class LatencyTimer:
    """
    Usage:
        with LatencyTimer("signal_generation_ms", METRICS):
            ...your code...
    """

    def __init__(self, metric_name: str, metrics: Metrics) -> None:
        self._name = metric_name
        self._metrics = metrics
        self._start: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._metrics.observe(self._name, elapsed_ms)
