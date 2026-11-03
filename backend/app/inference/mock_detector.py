"""Deterministic scripted detector for tests and GPU-less demo mode (`models.backend: mock`). Clearly not a model."""

from __future__ import annotations

from collections.abc import Callable

from app.inference.types import Detection, FrameDetections
from app.streaming.frame import Frame

VERSION = "mock@scripted"


def person_with_helmet(x: float, y: float, w: float, h: float) -> list[Detection]:
    hw = 0.5 * w
    return [Detection("person", 0.9, (x, y, x + w, y + h), VERSION),
            Detection("helmet", 0.88, (x + (w - hw) / 2, y + 0.01 * h, x + (w + hw) / 2, y + 0.12 * h), VERSION)]


def person_without_helmet(x: float, y: float, w: float, h: float) -> list[Detection]:
    hw = 0.42 * w
    return [Detection("person", 0.9, (x, y, x + w, y + h), VERSION),
            Detection("head", 0.86, (x + (w - hw) / 2, y + 0.02 * h, x + (w + hw) / 2, y + 0.15 * h), VERSION)]


def default_script(frame: Frame) -> list[Detection]:
    """Odd camera numbers: one worker without a helmet walking right; even: one worker with a helmet."""
    digits = "".join(ch for ch in frame.camera_id if ch.isdigit()) or "0"
    x = 100 + (frame.frame_id * 4) % max(1, frame.width - 300)
    make = person_without_helmet if int(digits) % 2 else person_with_helmet
    return make(x, frame.height * 0.2, frame.width * 0.1, frame.height * 0.6)


class MockDetector:
    name = "mock"
    version = VERSION

    def __init__(self, script: Callable[[Frame], list[Detection]] | None = None):
        self.script = script or default_script

    def model_versions(self) -> dict[str, str]:
        return {"person": VERSION, "ppe": VERSION}

    def warmup(self) -> None:
        return None

    def detect(self, frames: list[Frame]) -> list[FrameDetections]:
        return [FrameDetections(f, self.script(f), {"person": 0.0, "ppe": 0.0}) for f in frames]
