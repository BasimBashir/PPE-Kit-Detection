"""Per-camera person tracking (FR-TRK-001..003, ADR-004).

Uses Ultralytics' BYTETracker directly (one instance per camera). Ultralytics `model.track(persist=True)` is NOT used:
its tracker state is bound to one predictor and would mix cameras inside a batched predict call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Protocol

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from app.config.settings import TrackingConfig
from app.inference.types import Detection
from app.ppe.types import TrackedPerson
from app.streaming.frame import Frame


class PersonTracker(Protocol):
    def update(self, persons: list[Detection], frame: Frame) -> list[TrackedPerson]: ...

    def reset(self) -> None: ...

    @property
    def active_count(self) -> int: ...


class _Dets:
    """Minimal numpy-backed detections object accepted by BYTETracker.update (conf, cls, xywh, bool indexing)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf = xyxy, conf
        self.cls = np.zeros(len(conf), np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> _Dets:
        return _Dets(self.xyxy[idx], self.conf[idx])


class ByteTrackAdapter:
    def __init__(self, cfg: TrackingConfig, frame_rate: float):
        self.cfg, self.frame_rate = cfg, frame_rate
        self._new()

    def _new(self) -> None:
        args = SimpleNamespace(
            track_high_thresh=self.cfg.track_activation_threshold,
            track_low_thresh=0.1,
            new_track_thresh=self.cfg.track_activation_threshold,
            track_buffer=max(1, round(self.cfg.lost_track_buffer_s * self.frame_rate)),
            match_thresh=self.cfg.minimum_matching_threshold,
            fuse_score=True,
        )
        self._t = BYTETracker(args)  # track_buffer is in processed frames, hence the * frame_rate above
        self._active = 0

    def reset(self) -> None:
        self._new()

    @property
    def active_count(self) -> int:
        return self._active

    def update(self, persons: list[Detection], frame: Frame) -> list[TrackedPerson]:
        if persons:
            xyxy = np.array([p.xyxy for p in persons], np.float32)
            conf = np.array([p.conf for p in persons], np.float32)
        else:
            xyxy, conf = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        out = self._t.update(_Dets(xyxy, conf), frame.image)
        tracked = []
        for row in out:
            x1, y1, x2, y2, tid, score, _cls, idx = row[:8]
            src = persons[int(idx)]
            tracked.append(TrackedPerson(int(tid), (float(x1), float(y1), float(x2), float(y2)), float(score),
                                         src.keypoints))
        self._active = len(tracked)
        return tracked


def make_tracker(cfg: TrackingConfig, frame_rate: float) -> PersonTracker:
    return ByteTrackAdapter(cfg, frame_rate)
