"""Frame contract shared by ingest, scheduler and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Frame:
    camera_id: str
    frame_id: int  # monotonically increasing per camera source instance
    image: np.ndarray  # HxWx3 uint8 BGR
    source_pts_s: float | None  # stream presentation timestamp (seconds) if known
    captured_at: float  # host wall clock (time.time()) when decoded - used for persisted timestamps
    captured_mono: float  # time.monotonic() when decoded - used for latency/age math

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])
