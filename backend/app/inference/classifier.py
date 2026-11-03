"""Optional head-crop classifier (FR-DET-006): HELMET_PRESENT / HELMET_ABSENT / UNCERTAIN."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.config.settings import ModelSpec
from app.inference.types import HeadOutcome
from app.inference.ultralytics_detector import resolve_weights, weights_version
from app.observability import metrics as m


class UltralyticsHeadClassifier:
    def __init__(self, spec: ModelSpec, models_dir: Path, uncertain_below: float):
        from ultralytics import YOLO

        self.spec, self.uncertain_below = spec, uncertain_below
        path = resolve_weights(spec.weights, models_dir)
        self.version = weights_version(path)
        self.model = YOLO(str(path))
        names = {int(k): v for k, v in self.model.names.items()}
        self._map: dict[int, HeadOutcome] = {
            k: ("HELMET_PRESENT" if v == "helmet" else "HELMET_ABSENT") for k, v in names.items()
        }

    def classify(self, crops: list[np.ndarray]) -> list[tuple[HeadOutcome, float]]:
        out: list[tuple[HeadOutcome, float]] = [("UNCERTAIN", 0.0)] * len(crops)  # one result per input, always
        idx = [i for i, c in enumerate(crops) if c.size]
        if not idx:
            return out
        t0 = time.perf_counter()
        res = self.model.predict([crops[i] for i in idx], imgsz=self.spec.imgsz,
                                 quantize=16 if self.spec.half and self.spec.device != "cpu" else 32,
                                 device=self.spec.device, verbose=False)
        m.inference_duration_seconds.labels("classifier").observe(time.perf_counter() - t0)
        m.inference_requests_total.labels("classifier").inc()
        m.ppe_classifier_invocations_total.inc(len(idx))
        for i, r in zip(idx, res, strict=True):
            p = float(r.probs.top1conf)
            out[i] = (self._map[int(r.probs.top1)] if p >= self.uncertain_below else "UNCERTAIN", p)
        return out
