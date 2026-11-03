"""Evidence snapshots on local disk (FR-ALT-003, NFR-PRIV-002, ADR-008). Never raises into the pipeline."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import EvidenceConfig
from app.observability import metrics as m

log = logging.getLogger(__name__)


class EvidenceStore:
    def __init__(self, root: Path, cfg: EvidenceConfig, disk_usage: Callable[[str], object] = shutil.disk_usage):
        self.root = root.resolve()
        self.cfg = cfg
        self.disk_usage = disk_usage
        self.root.mkdir(parents=True, exist_ok=True)

    def free_bytes(self) -> int:
        try:
            return int(self.disk_usage(str(self.root)).free)  # type: ignore[attr-defined]
        except OSError:
            return 0

    def save(self, event_id: uuid.UUID, image_bgr: np.ndarray, captured_at: float) -> tuple[str | None, str | None]:
        """Returns (relative posix path, error). Opaque filename = event uuid (no names / identities)."""
        if self.free_bytes() < self.cfg.min_free_bytes:
            m.ppe_evidence_failures_total.labels("disk_low").inc()
            log.warning("evidence skipped for %s: free space below %d bytes", event_id, self.cfg.min_free_bytes)
            return None, "disk_low"
        day = datetime.fromtimestamp(captured_at, UTC)
        rel = f"{day:%Y/%m/%d}/{event_id}.jpg"
        dst = self.root / rel
        try:
            ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if not ok:
                raise OSError("jpeg encode failed")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(buf.tobytes())
            tmp.replace(dst)
        except OSError as e:
            m.ppe_evidence_failures_total.labels("write_failed").inc()
            log.error("evidence write failed for %s: %s", event_id, e)
            return None, "write_failed"
        return rel, None

    def open(self, rel_path: str) -> Path:
        """Resolve a stored relative path strictly inside the evidence root (rejects traversal)."""
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            raise FileNotFoundError("evidence not found")
        return p

    def delete_older_than(self, days: int) -> tuple[int, int]:
        cutoff = time.time() - days * 86400
        files = freed = 0
        for p in self.root.rglob("*.jpg"):
            try:
                st = p.stat()
                if st.st_mtime < cutoff:
                    p.unlink()
                    files, freed = files + 1, freed + st.st_size
            except OSError:
                continue
        for d in sorted((d for d in self.root.rglob("*") if d.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
        return files, freed
