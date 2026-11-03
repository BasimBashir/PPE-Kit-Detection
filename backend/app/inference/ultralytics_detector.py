"""Ultralytics YOLO26 cascade: pose person stage -> helmet/head stage (FR-DET-001/002/005/009, ADR-003)."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from app.config.settings import ModelsConfig, ModelSpec
from app.inference.types import Detection, FrameDetections
from app.observability import metrics as m
from app.streaming.frame import Frame

log = logging.getLogger(__name__)


def resolve_weights(weights: str, models_dir: Path) -> Path:
    """models_dir/weights if present; else download a known Ultralytics asset INTO models_dir (offline afterwards)."""
    p = Path(weights)
    if p.is_absolute() and p.exists():
        return p
    local = models_dir / weights
    if local.exists():
        return local
    from ultralytics.utils.downloads import attempt_download_asset

    models_dir.mkdir(parents=True, exist_ok=True)
    got = Path(attempt_download_asset(local))
    if not got.exists():
        raise FileNotFoundError(f"model weights not found: {local} (see models/README.md)")
    return got


def weights_version(path: Path) -> str:
    return f"{path.stem}@{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


class _Stage:
    def __init__(self, spec: ModelSpec, models_dir: Path):
        from ultralytics import YOLO

        self.spec = spec
        self.path = resolve_weights(spec.weights, models_dir)
        self.version = weights_version(self.path)
        self.model = YOLO(str(self.path))
        self.half = spec.half and spec.device != "cpu"

    def predict(self, images: list[np.ndarray]):
        return self.model.predict(images, imgsz=self.spec.imgsz, conf=self.spec.conf, quantize=16 if self.half else 32,
                                  device=self.spec.device, verbose=False)


class UltralyticsCascadeDetector:
    name = "ultralytics-cascade"

    def __init__(self, models: ModelsConfig, models_dir: Path):
        self.cfg = models
        self.person = _Stage(models.person, models_dir)
        self.ppe = _Stage(models.ppe, models_dir)
        self._ppe_ids = {v: k for k, v in models.ppe_class_map.items()}  # class index -> "helmet" | "head"
        self.version = f"{self.person.version}+{self.ppe.version}"

    def model_versions(self) -> dict[str, str]:
        return {"person": self.person.version, "ppe": self.ppe.version}

    def warmup(self) -> None:
        dummy = [np.zeros((720, 1280, 3), np.uint8)]
        self.person.predict(dummy)
        self.ppe.predict(dummy)

    def _timed(self, stage: str, fn, images):
        t0 = time.perf_counter()
        try:
            out = fn(images)
        except Exception:
            m.inference_errors_total.labels(stage).inc()
            raise
        dt = time.perf_counter() - t0
        m.inference_duration_seconds.labels(stage).observe(dt)
        m.inference_requests_total.labels(stage).inc()
        return out, dt

    def detect(self, frames: list[Frame]) -> list[FrameDetections]:
        if not frames:
            return []
        m.inference_batch_size.observe(len(frames))
        images = [f.image for f in frames]
        res, dt = self._timed("person", self.person.predict, images)
        out = [FrameDetections(f, stage_ms={"person": 1000 * dt / len(frames)}) for f in frames]
        for fd, r in zip(out, res, strict=True):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
            kps = r.keypoints.data.cpu().numpy().astype(np.float32) if r.keypoints is not None else None
            for i in np.flatnonzero(cls == 0):
                fd.detections.append(Detection("person", float(conf[i]), tuple(map(float, xyxy[i])),
                                               self.person.version, None if kps is None else kps[i]))
            m.detections_total.labels("person", self.person.version).inc(int((cls == 0).sum()))

        idx = [i for i, fd in enumerate(out) if fd.detections or not self.cfg.cascade_skip_ppe_without_person]
        if idx:
            res2, dt2 = self._timed("ppe", self.ppe.predict, [images[i] for i in idx])
            for i, r in zip(idx, res2, strict=True):
                out[i].stage_ms["ppe"] = 1000 * dt2 / len(idx)
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
                for j in range(len(cls)):
                    name = self._ppe_ids.get(int(cls[j]))
                    if name:
                        out[i].detections.append(Detection(name, float(conf[j]), tuple(map(float, xyxy[j])),
                                                           self.ppe.version))
                        m.detections_total.labels(name, self.ppe.version).inc()
        return out
