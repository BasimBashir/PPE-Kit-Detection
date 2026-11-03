"""Fair, FPS-paced batch scheduler over per-camera latest-frame slots (implementation-plan 4.4)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from app.streaming.frame import Frame
from app.streaming.slot import LatestFrameSlot


class InferenceScheduler:
    def __init__(self, slots: dict[str, LatestFrameSlot], target_fps: dict[str, float], max_batch: int,
                 max_batch_wait_ms: float, cond: threading.Condition, clock: Callable[[], float] = time.monotonic):
        self.slots, self.target_fps = dict(slots), dict(target_fps)
        self.max_batch, self.max_wait = max_batch, max_batch_wait_ms / 1000.0
        self.cond, self.clock = cond, clock
        now = clock()
        self.next_due = {c: now for c in self.slots}
        self.last_served = {c: float("-inf") for c in self.slots}  # never served sorts first

    def add_camera(self, camera_id: str, slot: LatestFrameSlot, fps: float) -> None:
        with self.cond:
            self.slots[camera_id], self.target_fps[camera_id] = slot, fps
            self.next_due[camera_id], self.last_served[camera_id] = self.clock(), float("-inf")

    def remove_camera(self, camera_id: str) -> None:
        with self.cond:
            for d in (self.slots, self.target_fps, self.next_due, self.last_served):
                d.pop(camera_id, None)

    def set_target_fps(self, camera_id: str, fps: float) -> None:
        with self.cond:
            self.target_fps[camera_id] = fps

    def _due(self, now: float) -> list[str]:
        due = [c for c, s in self.slots.items() if now >= self.next_due[c] and s.peek_new()]
        return sorted(due, key=lambda c: self.last_served[c])  # least recently served first

    def _earliest_due(self, now: float) -> float:
        pending = [self.next_due[c] for c, s in self.slots.items() if s.peek_new()]
        return max(0.0, min(pending) - now) if pending else float("inf")

    def next_batch(self, timeout: float) -> list[Frame]:
        deadline = time.monotonic() + timeout  # waits are real time even when `clock` is injected (tests)
        with self.cond:
            while True:
                now = self.clock()
                due = self._due(now)
                if due and self.max_wait > 0 and len(due) < self.max_batch:
                    # Optionally wait a little for more cameras so the GPU batch is fuller (bounded).
                    wait_until = min(deadline, time.monotonic() + self.max_wait)
                    while len(due) < self.max_batch and time.monotonic() < wait_until:
                        self.cond.wait(max(0.0, wait_until - time.monotonic()))
                        due = self._due(self.clock())
                if due:
                    batch: list[Frame] = []
                    now = self.clock()
                    for c in due[: self.max_batch]:
                        f = self.slots[c].take()
                        if f is None:
                            continue
                        # Keep cadence, but never accumulate "owed" frames after a stall or overload.
                        period = 1.0 / self.target_fps[c]
                        nd = self.next_due[c] + period
                        self.next_due[c] = nd if nd >= now else now + period
                        self.last_served[c] = now
                        batch.append(f)
                    if batch:
                        return batch
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self.cond.wait(min(remaining, self._earliest_due(self.clock()) or remaining))
