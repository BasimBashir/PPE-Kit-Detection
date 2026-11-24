"""Face detection + embedding for crossing identity: YuNet detector, SFace recogniser (ADR-016).

Both models come from the OpenCV model zoo (MIT / Apache-2.0) and are downloaded into the models dir on first use,
like the Ultralytics weights - nothing else is added to the dependency list. SFace is trained on frontal-ish faces, so
the caller gets a yaw/pitch estimate per face and matches with angle-dependent thresholds (see identity.py).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ppe.types import Box

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
DETECTOR = ("face_detection_yunet_2023mar.onnx", f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx")
RECOGNISER = ("face_recognition_sface_2021dec.onnx",
              f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx")

FRONTAL, QUARTER, PROFILE = "frontal", "quarter", "profile"


@dataclass(slots=True)
class Face:
    image: np.ndarray  # aligned 112x112 crop (what the recogniser consumes)
    score: float
    angle: str  # frontal | quarter | profile, from the landmarks


def _ensure(name: str, url: str, models_dir: Path) -> Path:
    path = models_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    urllib.request.urlretrieve(url, part)  # noqa: S310 - fixed https URL
    os.replace(part, path)
    return path


def angle_of(yaw: float, pitch: float) -> str:
    """Landmark-derived view bucket: |yaw| (nose vs eye midpoint) decides, |pitch| demotes a frontal to quarter."""
    if abs(yaw) >= 0.6:
        return PROFILE
    return FRONTAL if abs(yaw) < 0.25 and abs(pitch) < 0.5 else QUARTER


class FaceEngine:
    """Lazy OpenCV DNN wrapper. Detects only inside a person box when one is given (a door frame has many faces)."""

    def __init__(self, models_dir: Path, score: float = 0.8):
        self.models_dir, self.score = models_dir, score
        self._det: cv2.FaceDetectorYN | None = None
        self._rec: cv2.FaceRecognizerSF | None = None
        self._size: tuple[int, int] = (0, 0)

    def _load(self) -> None:
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(str(_ensure(*DETECTOR, self.models_dir)), "", (320, 320),
                                                  score_threshold=self.score)
            self._rec = cv2.FaceRecognizerSF.create(str(_ensure(*RECOGNISER, self.models_dir)), "")

    def faces(self, image: np.ndarray, box: Box | None = None) -> list[Face]:
        """Aligned faces on `image`, or inside `box` (xyxy pixel coords, clamped); at most the best one per box."""
        self._load()
        ox, oy = 0, 0
        if box is not None:
            h, w = image.shape[:2]
            x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
            x1, y1 = min(w, int(box[2])), min(h, int(box[3]))
            if x1 - x0 < 16 or y1 - y0 < 16:
                return []
            image, ox, oy = image[y0:y1, x0:x1], x0, y0
        h, w = image.shape[:2]
        assert self._det is not None and self._rec is not None
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self._det.detect(image)
        out: list[Face] = []
        for r in rows if rows is not None else ():
            rx, ry, lx, ly = (float(r[i]) for i in (4, 5, 6, 7))
            nx, ny = float(r[8]), float(r[9])
            d = float(np.hypot(lx - rx, ly - ry)) or 1.0
            row = r.copy()
            row[[0, 4, 6, 8, 10, 12]] += ox  # box x + landmark x's back to frame coords ...
            row[[1, 5, 7, 9, 11, 13]] += oy  # ... (w/h in row[2:4] are already absolute)
            out.append(Face(self._rec.alignCrop(image, row), float(r[14]),
                            angle_of((nx - (rx + lx) / 2) / d, (ny - (ry + ly) / 2) / d)))
        if box is not None and out:
            return [max(out, key=lambda f: f.score)]  # the person's own face, not a neighbour's in the padded box
        return out

    def embed(self, face: Face) -> list[float]:
        """Unit-length 128-d SFace descriptor."""
        assert self._rec is not None
        v = self._rec.feature(face.image).ravel().astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n).tolist() if n else []
