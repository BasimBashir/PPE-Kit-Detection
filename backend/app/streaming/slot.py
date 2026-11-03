"""Bounded latest-frame slot: depth 0..1, newest frame wins (NFR-PERF-006, spec 11.3)."""

from __future__ import annotations

import threading

from app.observability import metrics as m
from app.streaming.frame import Frame


class LatestFrameSlot:
    def __init__(self, camera_id: str, cond: threading.Condition):
        self.camera_id = camera_id
        self._cond = cond  # shared with the scheduler so a put wakes it up
        self._frame: Frame | None = None
        self.dropped = 0
        self._dropped_metric = m.camera_frames_dropped_total.labels(camera_id)
        self._depth_metric = m.camera_frame_queue_depth.labels(camera_id)

    def put(self, frame: Frame) -> bool:
        """Store `frame`; returns True when an unconsumed frame was overwritten (counted as dropped)."""
        with self._cond:
            overwritten = self._frame is not None
            if overwritten:
                self.dropped += 1
                self._dropped_metric.inc()
            self._frame = frame
            self._depth_metric.set(1)
            self._cond.notify_all()
        return overwritten

    def take(self) -> Frame | None:
        with self._cond:
            f, self._frame = self._frame, None
            self._depth_metric.set(0)
            return f

    def peek_new(self) -> bool:
        with self._cond:
            return self._frame is not None

    @property
    def depth(self) -> int:
        with self._cond:
            return 1 if self._frame is not None else 0
