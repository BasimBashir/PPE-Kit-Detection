"""Messages between the API process and the pipeline worker process (multiprocessing queues).

worker -> API (telemetry queue, bounded; previews are dropped when full, everything else blocks <=0.5 s):
  hello           once at start
  camera_status   every 1 s, all cameras
  preview         per camera at pipeline.preview_fps (JPEG + overlay metadata)
  event           immediately after an event is created/updated/spooled
  stats           every 1 s (inference + resource summary)
  heartbeat       every 1 s (supervisor liveness)
API -> worker (command queue):
  reload_cameras | reload_rules | shutdown | fault (only when PPE_ENABLE_FAULT_INJECTION=1)

Events are durable in PostgreSQL / the spool; the `event` message is only a low-latency nudge.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class CameraStatus(TypedDict):
    camera_id: str
    state: str  # one of metrics.CAMERA_STATES
    state_reason: str  # human-readable, already redacted
    state_since: float  # wall clock
    last_frame_at: float | None
    last_frame_age_s: float | None
    source_fps: float | None
    decoded_fps: float
    processed_fps: float
    target_fps: float
    reconnect_count: int
    decode_errors: int
    frames_dropped: int
    queue_depth: int
    frame_age_at_inference_p95_s: float | None  # rolling 60 s
    uptime_ratio: float  # ONLINE|DEGRADED seconds / seconds since worker start
    last_recovery_s: float | None
    active_tracks: int
    width: int | None
    height: int | None


class OverlayTrack(TypedDict):
    track_id: int
    box: list[float]  # x1,y1,x2,y2 in preview pixel space
    ppe_state: str
    outcome: str
    head_roi: list[float] | None
    helmet_box: list[float] | None
    head_box: list[float] | None


class OverlayDet(TypedDict):
    cls: str
    conf: float
    box: list[float]


class HelloMsg(TypedDict):
    type: Literal["hello"]
    run_id: str
    pid: int
    started_at: float
    model_versions: dict[str, str]
    device: str


class CameraStatusMsg(TypedDict):
    type: Literal["camera_status"]
    ts: float
    cameras: list[CameraStatus]


class PreviewMsg(TypedDict):
    type: Literal["preview"]
    camera_id: str
    frame_id: int
    captured_at: float
    width: int  # preview size
    height: int
    jpeg: bytes
    tracks: list[OverlayTrack]
    detections: list[OverlayDet]


class EventMsg(TypedDict):
    type: Literal["event"]
    action: Literal["created", "updated", "closed", "spooled"]
    event_id: str
    camera_id: str
    persisted_at: float | None


class StatsMsg(TypedDict):
    type: Literal["stats"]
    ts: float
    aggregate_processed_fps: float
    inference: dict[str, dict[str, float]]  # stage -> {p50_ms, p95_ms, p99_ms, count}
    batch_size_mean: float
    resources: dict[str, float | None]  # rss_bytes, cpu_percent, gpu_util, gpu_mem_used, gpu_mem_total, ...
    spool_pending: int
    db_ok: bool


class HeartbeatMsg(TypedDict):
    type: Literal["heartbeat"]
    ts: float
    run_id: str


class Command(TypedDict):
    type: Literal["reload_cameras", "reload_rules", "shutdown", "fault"]
    kind: NotRequired[str]  # fault kind: inference_exception | gpu_oom | crash | disk_full | decoder_error
    camera_id: NotRequired[str]
    duration_s: NotRequired[float]
