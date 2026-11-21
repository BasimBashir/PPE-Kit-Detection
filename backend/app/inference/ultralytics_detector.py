"""Ultralytics YOLO26 cascade: pose person stage -> helmet/head stage (FR-DET-001/002/005/009, ADR-003)."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import cv2
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
        # Phase 3: vest / gloves / goggles / boots (+ "no_*") detector; the helmet stack above stays as measured
        self.items = _Stage(models.items, models_dir) if models.items else None
        self._items_ids = {v: k for k, v in models.items_class_map.items()}
        self.version = f"{self.person.version}+{self.ppe.version}" + (f"+{self.items.version}" if self.items else "")
        # Phase 4: COCO detector for phones (ADR-015); same downscaled frames, one more batched call
        self.phone = _Stage(models.phone, models_dir) if models.phone else None
        self._phone_ids = {v: k for k, v in models.phone_class_map.items()}
        if self.phone:
            self.version += f"+{self.phone.version}"

    def model_versions(self) -> dict[str, str]:
        return {"person": self.person.version, "ppe": self.ppe.version,
                **({"items": self.items.version} if self.items else {}),
                **({"phone": self.phone.version} if self.phone else {})}

    def warmup(self) -> None:
        dummy = [np.zeros((720, 1280, 3), np.uint8)]
        for st in (self.person, self.ppe, self.items, self.phone):
            if st:
                st.predict(dummy)

    def _fit(self, frames: list[Frame]) -> tuple[list[np.ndarray], list[float]]:
        """Images no larger than the biggest stage input (long side), and the factor back to source pixels."""
        side = max(st.spec.imgsz for st in (self.person, self.ppe, self.items, self.phone) if st)
        images, k = [], []
        for f in frames:
            h, w = f.image.shape[:2]
            s = side / max(h, w)
            if s < 1.0:
                images.append(cv2.resize(f.image, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR))
                k.append(max(h, w) / side)
            else:
                images.append(f.image)
                k.append(1.0)
        return images, k

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
        # 4K: downscale once to the largest model input instead of letterboxing full frames once per stage (CPU-bound,
        # docs/benchmarks/phase3-4k.md). Models see the same pixels; boxes map back to full-res for crops/evidence.
        images, k = self._fit(frames)
        res, dt = self._timed("person", self.person.predict, images)
        out = [FrameDetections(f, stage_ms={"person": 1000 * dt / len(frames)}) for f in frames]
        for fi, (fd, r) in enumerate(zip(out, res, strict=True)):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy, conf, cls = b.xyxy.cpu().numpy() * k[fi], b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
            kps = r.keypoints.data.cpu().numpy().astype(np.float32) if r.keypoints is not None else None
            if kps is not None:
                kps[..., :2] *= k[fi]
            for i in np.flatnonzero(cls == 0):
                fd.detections.append(Detection("person", float(conf[i]), tuple(map(float, xyxy[i])),
                                               self.person.version, None if kps is None else kps[i]))
            m.detections_total.labels("person", self.person.version).inc(int((cls == 0).sum()))

        idx = [i for i, fd in enumerate(out) if fd.detections or not self.cfg.cascade_skip_ppe_without_person]
        if idx:
            for stage, st, ids in (("ppe", self.ppe, self._ppe_ids), ("items", self.items, self._items_ids),
                                   ("phone", self.phone, self._phone_ids)):
                if st is None:
                    continue
                res2, dt2 = self._timed(stage, st.predict, [images[i] for i in idx])
                for i, r in zip(idx, res2, strict=True):
                    out[i].stage_ms[stage] = 1000 * dt2 / len(idx)
                    b = r.boxes
                    if b is None or len(b) == 0:
                        continue
                    xyxy, conf, cls = b.xyxy.cpu().numpy() * k[i], b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
                    for j in range(len(cls)):
                        name = ids.get(int(cls[j]))
                        if name:
                            out[i].detections.append(Detection(name, float(conf[j]), tuple(map(float, xyxy[j])),
                                                               st.version))
                            m.detections_total.labels(name, st.version).inc()
        return out
