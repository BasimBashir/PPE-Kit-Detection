"""Small rolling-window statistics for the JSON metrics summary (Prometheus histograms remain the source of truth)."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np


class RollingWindow:
    """Values observed within the last `window_s` seconds; thread-safe; bounded by `max_items`."""

    def __init__(self, window_s: float = 60.0, max_items: int = 20000):
        self.window_s = window_s
        self._d: deque[tuple[float, float]] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def add(self, value: float, now: float | None = None) -> None:
        with self._lock:
            self._d.append((now if now is not None else time.monotonic(), value))

    def _values(self, now: float) -> list[float]:
        while self._d and now - self._d[0][0] > self.window_s:
            self._d.popleft()
        return [v for _, v in self._d]

    def count(self, now: float | None = None) -> int:
        with self._lock:
            return len(self._values(now if now is not None else time.monotonic()))

    def quantiles(self, now: float | None = None) -> dict[str, float] | None:
        with self._lock:
            vals = self._values(now if now is not None else time.monotonic())
        if not vals:
            return None
        p50, p95, p99 = np.percentile(np.asarray(vals), [50, 95, 99])
        return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "count": len(vals)}

    def rate(self, now: float | None = None) -> float:
        """Events per second over the window (for FPS counters, add 1.0 per event)."""
        return self.count(now) / self.window_s
