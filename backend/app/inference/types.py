"""Detection contracts (FR-DET-001/002/009)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from app.streaming.frame import Frame

DetClass = Literal["person", "helmet", "head"]

# COCO-17 keypoint indices used for head ROI
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR, KP_LSHOULDER, KP_RSHOULDER = 0, 1, 2, 3, 4, 5, 6


@dataclass(slots=True)
class Detection:
    cls: DetClass
    conf: float
    xyxy: tuple[float, float, float, float]  # pixel coords in the source frame
    model_version: str
    keypoints: np.ndarray | None = None  # (17, 3) x, y, conf - person detections from a pose model only


@dataclass(slots=True)
class FrameDetections:
    frame: Frame
    detections: list[Detection] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)  # e.g. {"person": 7.1, "ppe": 6.3}


class Detector(Protocol):
    """Batch detector. Must return one FrameDetections per input frame, same order."""

    name: str
    version: str

    def warmup(self) -> None: ...

    def detect(self, frames: list[Frame]) -> list[FrameDetections]: ...


HeadOutcome = Literal["HELMET_PRESENT", "HELMET_ABSENT", "UNCERTAIN"]


class HeadClassifier(Protocol):
    """FR-DET-006: classify head crops (BGR uint8). Returns (outcome, confidence) per crop."""

    version: str

    def classify(self, crops: list[np.ndarray]) -> list[tuple[HeadOutcome, float]]: ...
