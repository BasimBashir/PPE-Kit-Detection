"""Prometheus metric definitions (spec 10.1). Single source of metric names - deployments/docker/alert-rules.yml
and the Grafana dashboard depend on them.

Two processes export metrics: the API process (port 8000 /metrics) and the pipeline worker (port 9101).
Each process only updates the metrics it owns; unused ones simply stay at zero / absent.
Label rules: camera_id / stage / class / channel / state only - never track or event ids (bounded cardinality).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LAT_BUCKETS = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2, 5)
AGE_BUCKETS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)
EVENT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30)

CAMERA_STATES = ("DISABLED", "CONNECTING", "ONLINE", "DEGRADED", "OFFLINE", "RECOVERING", "ERROR")

# ---------------------------------------------------------------- per camera (worker)
camera_connection_state = Gauge(
    "camera_connection_state", "1 for the camera's current lifecycle state, 0 otherwise", ["camera_id", "state"]
)
camera_last_frame_age_seconds = Gauge("camera_last_frame_age_seconds", "Seconds since last decoded frame", ["camera_id"])
camera_source_fps = Gauge("camera_source_fps", "Nominal stream FPS reported by the source", ["camera_id"])
camera_decoded_fps = Gauge("camera_decoded_fps", "Measured decode FPS (1 s window)", ["camera_id"])
camera_processed_fps = Gauge("camera_processed_fps", "Measured inference FPS (5 s window)", ["camera_id"])
camera_target_fps = Gauge("camera_target_fps", "Configured processed FPS target", ["camera_id"])
camera_reconnect_total = Counter("camera_reconnect_total", "Reconnect attempts", ["camera_id"])
camera_decode_errors_total = Counter("camera_decode_errors_total", "Decode/read errors", ["camera_id"])
camera_frames_dropped_total = Counter(
    "camera_frames_dropped_total", "Frames overwritten in the latest-frame slot before inference took them", ["camera_id"]
)
camera_frame_queue_depth = Gauge("camera_frame_queue_depth", "Frames waiting in the camera slot (0..1)", ["camera_id"])
camera_frame_age_at_inference_seconds = Histogram(
    "camera_frame_age_at_inference_seconds", "Decode-to-inference-start age", ["camera_id"], buckets=AGE_BUCKETS
)
camera_online_seconds_total = Counter(
    "camera_online_seconds_total", "Seconds spent ONLINE or DEGRADED (uptime numerator)", ["camera_id"]
)
camera_last_recovery_seconds = Gauge(
    "camera_last_recovery_seconds", "Duration of the last outage (offline -> first frame)", ["camera_id"]
)

# ---------------------------------------------------------------- inference (worker)
inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Per-batch stage latency; stage=person|ppe|classifier|tracking|association|total",
    ["stage"],
    buckets=LAT_BUCKETS,
)
inference_batch_size = Histogram("inference_batch_size", "Frames per inference batch", buckets=(1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32))
inference_requests_total = Counter("inference_requests_total", "Batches executed per stage", ["stage"])
inference_errors_total = Counter("inference_errors_total", "Failed batches per stage", ["stage"])
detections_total = Counter("detections_total", "Detections above threshold", ["class", "model"])
tracker_active_tracks = Gauge("tracker_active_tracks", "Active person tracks", ["camera_id"])
ppe_state_transitions_total = Counter("ppe_state_transitions_total", "PPE FSM transitions", ["from_state", "to_state"])
ppe_uncertain_observations_total = Counter(
    "ppe_uncertain_observations_total", "Observations with UNCERTAIN outcome", ["camera_id"]
)
ppe_classifier_invocations_total = Counter("ppe_classifier_invocations_total", "Head crops sent to the classifier")

# ---------------------------------------------------------------- events / alerts
ppe_events_created_total = Counter("ppe_events_created_total", "New violation events", ["camera_id"])
ppe_events_deduplicated_total = Counter(
    "ppe_events_deduplicated_total", "Confirmations merged into an existing event", ["reason"]
)
ppe_events_spooled_total = Counter("ppe_events_spooled_total", "Events written to the local spool (DB down)")
ppe_spool_pending = Gauge("ppe_spool_pending", "Spooled events awaiting replay")
ppe_evidence_failures_total = Counter("ppe_evidence_failures_total", "Evidence snapshot failures", ["reason"])
ppe_event_confirmation_latency_seconds = Histogram(
    "ppe_event_confirmation_latency_seconds",
    "Confirming frame decoded -> event persisted (host-ingest-to-persisted)",
    buckets=EVENT_BUCKETS,
)
ppe_event_time_to_confirm_seconds = Histogram(
    "ppe_event_time_to_confirm_seconds",
    "First violating observation -> confirmation (dominated by the temporal policy)",
    buckets=EVENT_BUCKETS,
)
ppe_alert_delivery_attempts_total = Counter("ppe_alert_delivery_attempts_total", "Outbox delivery attempts", ["channel"])
ppe_alert_delivery_failures_total = Counter("ppe_alert_delivery_failures_total", "Outbox delivery failures", ["channel"])
ppe_alert_outbox_pending = Gauge("ppe_alert_outbox_pending", "Undelivered outbox rows")
ppe_event_to_dashboard_latency_seconds = Histogram(
    "ppe_event_to_dashboard_latency_seconds",
    "Event persisted -> rendered in a dashboard client (client ack)",
    buckets=EVENT_BUCKETS,
)
ppe_websocket_clients = Gauge("ppe_websocket_clients", "Connected dashboard websocket clients")

# ---------------------------------------------------------------- resources / health
app_process_rss_bytes = Gauge("app_process_rss_bytes", "Resident memory", ["process"])
app_process_cpu_percent = Gauge("app_process_cpu_percent", "Process CPU percent (100 = one core)", ["process"])
system_cpu_percent = Gauge("system_cpu_percent", "Host CPU utilisation percent")
system_memory_used_bytes = Gauge("system_memory_used_bytes", "Host memory used")
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilisation", ["gpu"])
gpu_memory_used_bytes = Gauge("gpu_memory_used_bytes", "GPU memory used (whole device)", ["gpu"])
gpu_memory_total_bytes = Gauge("gpu_memory_total_bytes", "GPU memory total", ["gpu"])
gpu_temperature_celsius = Gauge("gpu_temperature_celsius", "GPU temperature", ["gpu"])
gpu_power_watts = Gauge("gpu_power_watts", "GPU power draw", ["gpu"])
torch_cuda_memory_allocated_bytes = Gauge("torch_cuda_memory_allocated_bytes", "PyTorch allocated CUDA memory (worker)")
disk_free_bytes = Gauge("disk_free_bytes", "Free bytes on the volume", ["path"])
db_up = Gauge("db_up", "1 if the database answered the last health probe")
ppe_worker_up = Gauge("ppe_worker_up", "1 while the pipeline worker process is alive and sending heartbeats")
ppe_worker_restarts_total = Counter("ppe_worker_restarts_total", "Pipeline worker restarts by the supervisor")
ppe_worker_uptime_seconds = Gauge("ppe_worker_uptime_seconds", "Seconds since the current worker started")
