"""In-memory live view fed by worker telemetry (API process). Durable truth stays in PostgreSQL."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from app.observability.stats import RollingWindow


def iso(wall: float | None) -> str | None:
    return None if wall is None else datetime.fromtimestamp(wall, UTC).isoformat().replace("+00:00", "Z")


class LiveState:
    HISTORY_STEP_S = 5.0
    HISTORY_POINTS = 120  # 10 minutes

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cameras: dict[str, dict[str, Any]] = {}
        self.previews: dict[str, dict[str, Any]] = {}
        self.stats: dict[str, Any] | None = None
        self.worker: dict[str, Any] = {"up": False, "pid": None, "run_id": None, "started_at": None,
                                       "restarts": 0, "device": None, "model_versions": {}}
        self.last_heartbeat = 0.0
        self.history: deque[dict[str, Any]] = deque(maxlen=self.HISTORY_POINTS)
        self._last_hist = 0.0
        self.persist_to_dashboard = RollingWindow(600.0)
        self.db_up = True
        self.started_at = time.time()

    # ------------------------------------------------------------ writers (telemetry thread)
    def apply(self, msg: dict[str, Any]) -> None:
        t = msg.get("type")
        with self._lock:
            if t in ("camera_status", "preview", "stats") and not self.worker["up"]:
                return  # buffered telemetry from a dead worker must never resurrect cameras as ONLINE
            if t == "camera_status":
                self.cameras = {c["camera_id"]: c for c in msg["cameras"]}
            elif t == "preview":
                self.previews[msg["camera_id"]] = msg
            elif t == "stats":
                self.stats = msg
                if time.time() - self._last_hist >= self.HISTORY_STEP_S:
                    self._last_hist = time.time()
                    r = msg.get("resources") or {}
                    p95s = [c["frame_age_at_inference_p95_s"] for c in self.cameras.values()
                            if c.get("frame_age_at_inference_p95_s") is not None]
                    self.history.append({"ts": iso(msg["ts"]), "processed_fps": msg["aggregate_processed_fps"],
                                         "gpu_util_percent": r.get("gpu_util_percent"),
                                         "worker_rss_bytes": r.get("rss_bytes"),
                                         "frame_age_p95_s": max(p95s) if p95s else None})
            elif t == "hello":
                self.worker.update(up=True, pid=msg["pid"], run_id=msg["run_id"], started_at=msg["started_at"],
                                   device=msg.get("device"), model_versions=msg.get("model_versions", {}))
                self.last_heartbeat = time.monotonic()
            elif t == "heartbeat":
                self.last_heartbeat = time.monotonic()

    def worker_down(self) -> None:
        with self._lock:
            self.worker.update(up=False, pid=None)
            self.cameras = {}  # never show stale ONLINE while the worker is gone
            self.previews = {}

    # ------------------------------------------------------------ readers
    def camera(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.cameras.get(camera_id)

    def all_cameras(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self.cameras)

    def preview(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.previews.get(camera_id)

    def snapshot_stats(self) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
        with self._lock:
            return self.stats, dict(self.worker), list(self.history)


def camera_status_out(c: dict[str, Any] | None) -> dict[str, Any] | None:
    """Telemetry CameraStatus -> API CameraStatus (ISO timestamps)."""
    if c is None:
        return None
    out = {k: v for k, v in c.items() if k != "camera_id"}
    out["state_since"] = iso(c.get("state_since"))
    out["last_frame_at"] = iso(c.get("last_frame_at"))
    return out
