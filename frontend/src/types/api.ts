// Port of docs/system-design/10-api-and-websocket-contracts.md - keep in sync with backend/app/schemas.

export type Role = "admin" | "operator" | "viewer";
export interface User { id: string; username: string; role: Role; enabled: boolean; created_at: string }

export type CameraState = "DISABLED" | "CONNECTING" | "ONLINE" | "DEGRADED" | "OFFLINE" | "RECOVERING" | "ERROR";
export const CAMERA_STATES: CameraState[] = ["DISABLED", "CONNECTING", "ONLINE", "DEGRADED", "OFFLINE", "RECOVERING", "ERROR"];

export interface CameraStatus {
  state: CameraState; state_reason: string; state_since: string | null; last_frame_at: string | null;
  last_frame_age_s: number | null; source_fps: number | null; decoded_fps: number; processed_fps: number;
  target_fps: number; reconnect_count: number; decode_errors: number; frames_dropped: number; queue_depth: number;
  frame_age_at_inference_p95_s: number | null; uptime_ratio: number; last_recovery_s: number | null;
  active_tracks: number; width: number | null; height: number | null;
}
export type CameraStatusMsgItem = CameraStatus & { camera_id: string };

export interface Camera {
  id: string; name: string; stream_uri: string; credential_ref: string | null; has_credentials: boolean;
  transport: "tcp" | "udp"; enabled: boolean; site: string | null; location: string | null; target_fps: number | null;
  created_at: string; updated_at: string; status: CameraStatus | null; open_violations: number;
}
export interface CameraCreate {
  id: string; name: string; stream_uri: string; credential_ref?: string | null; transport?: "tcp" | "udp";
  enabled?: boolean; site?: string | null; location?: string | null; target_fps?: number | null;
}
export type CameraUpdate = Partial<Omit<CameraCreate, "id">>;
export interface CameraTestResult {
  ok: boolean; latency_ms: number | null; codec: string | null; width: number | null; height: number | null;
  fps: number | null; error: string | null;
}

export interface Zone { id: string; camera_id: string; name: string; kind: "monitor" | "ignore" | "privacy"; polygon: [number, number][]; enabled: boolean }

export interface Thresholds { person_conf: number; helmet_conf: number; head_conf: number; min_person_height_px: number; min_head_px: number }
export interface TemporalPolicy {
  mode: "count" | "duration" | "both"; confirm_n: number; confirm_m: number; min_duration_s: number; suspect_n: number;
  compliant_n: number; compliant_m: number; clear_n: number; clear_m: number; unknown_ratio: number; window: number;
}
export interface Rule {
  id: string; name: string; ppe_class: "hard_hat"; camera_id: string | null; zone_id: string | null;
  severity: "low" | "medium" | "high"; enabled: boolean; thresholds: Thresholds; temporal_policy: TemporalPolicy;
  effective_confirm_seconds: number; updated_at: string;
}

export type EventStatus = "OPEN" | "ACKNOWLEDGED" | "RESOLVED";
export type Feedback = "TRUE_POSITIVE" | "FALSE_POSITIVE";
export interface PPEEvent {
  id: string; camera_id: string; camera_name: string; track_id: number; zone_id: string | null; rule_id: string;
  episode_id: string; status: EventStatus; severity: string; reason: string;
  confidence_summary: { no_helmet_votes: number; decisive_votes: number; mean_confidence: number; roi_source: string; classifier_used: boolean };
  first_seen_at: string; confirmed_at: string; frame_captured_at: string; persisted_at: string; last_seen_at: string;
  episode_ended_at: string | null; reconfirm_count: number; has_evidence: boolean; evidence_error: string | null;
  model_versions: Record<string, string>; feedback: Feedback | null; feedback_note: string | null;
  acknowledged_by: string | null; acknowledged_at: string | null; resolved_by: string | null; resolved_at: string | null;
  latency: { time_to_confirm_s: number | null; confirm_to_persist_s: number | null; frame_to_persist_s?: number | null; persist_to_dashboard_s: number | null };
}
export interface EventAction { id: number; action: string; actor: string; note: string | null; created_at: string }
export type EventDetail = PPEEvent & { actions: EventAction[] };
export interface Page<T> { items: T[]; total: number; limit: number; offset: number }

export interface StageStats { p50_ms: number; p95_ms: number; p99_ms: number; count: number }
export interface MetricsSummary {
  generated_at: string; mode: "demo" | "live";
  aggregate: { cameras_total: number; online: number; degraded: number; offline: number; error: number; disabled: number;
    processed_fps: number; target_fps_total: number; active_violations: number | null; events_last_hour: number | null };
  cameras: CameraStatusMsgItem[];
  inference: Record<string, StageStats>;
  batch_size_mean: number;
  latency: { frame_age_at_inference_p95_s: number | null; persist_to_dashboard_p95_s: number | null };
  resources: { api_rss_bytes: number | null; worker_rss_bytes: number | null; worker_cpu_percent: number | null;
    system_cpu_percent: number | null; system_memory_used_bytes: number | null; gpu_util_percent: number | null;
    gpu_memory_used_bytes: number | null; gpu_memory_total_bytes: number | null; gpu_temperature_c: number | null;
    gpu_power_w: number | null; torch_cuda_allocated_bytes: number | null; disk_free_bytes: number | null };
  worker: { up: boolean; pid: number | null; run_id: string | null; uptime_s: number | null; restarts: number; device: string | null };
  database: { up: boolean }; outbox: { pending: number; failed: number }; spool: { pending: number };
  history: { ts: string; processed_fps: number; gpu_util_percent: number | null; worker_rss_bytes: number | null; frame_age_p95_s: number | null }[];
}
export interface SettingsOut { evidence_retention_days: number; evidence_min_free_bytes: number }

export interface OverlayTrack { track_id: number; box: number[]; ppe_state: string; outcome: string; head_roi: number[] | null; helmet_box: number[] | null; head_box: number[] | null }
export interface OverlayDet { cls: "person" | "helmet" | "head"; conf: number; box: number[] }
export interface FrameOut { frame_id: number; captured_at: string; width: number; height: number; image_b64: string; tracks: OverlayTrack[]; detections: OverlayDet[] }

export type WsMessage =
  | { type: "snapshot"; cameras: Camera[]; open_events: PPEEvent[]; metrics: MetricsSummary }
  | { type: "camera_status"; cameras: CameraStatusMsgItem[] }
  | { type: "event_created"; event: PPEEvent }
  | { type: "event_updated"; event: PPEEvent }
  | { type: "metrics"; metrics: MetricsSummary }
  | { type: "worker"; up: boolean; pid: number | null; run_id: string | null }
  | { type: "ping"; ts: string };
