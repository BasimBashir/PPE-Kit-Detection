# Implementation plan and module contracts

Status: ACTIVE (2026-11-02). This file is the working contract for every implementer. The master spec is
[PPE_Intelligence_Master_Spec.md](../PPE_Intelligence_Master_Spec.md); progress lives in [BUILD_STATUS.md](../BUILD_STATUS.md).

## 1. Environment (inspected 2026-11-02)

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26300 |
| CPU / RAM | Intel i5-12400F (6C/12T) / 63.8 GB |
| GPU | NVIDIA RTX 3090 24 GB, driver 616.92 (CUDA 13.4 UMD) |
| Python | 3.12.10 via uv 0.12.5; venv outside OneDrive: `UV_PROJECT_ENVIRONMENT=C:/Users/mianx/.venvs/ppe-intelligence` |
| Torch | 2.14.1+cu130 (CUDA available, verified) |
| Node | 24.16.0 |
| ffmpeg | 8.1.2 (gyan.dev full build; `drawtext` unusable - no fontconfig) |
| Docker | 29.8.0 (Docker Desktop, WSL2) |

## 2. Pinned stack (versions queried from PyPI / Docker Hub 2026-11-02)

| Component | Version | License | Why |
|---|---|---|---|
| FastAPI / uvicorn | 0.162.0 / >=0.35 | MIT / BSD | API + WebSocket |
| SQLAlchemy / Alembic / psycopg | 2.1.6 / 1.20.1 / 3.3.6 | MIT / MIT / LGPL-3.0 | ORM, migrations, driver |
| PyAV | 19.0.2 | BSD-3 | RTSP/file decode (ADR-002) |
| Ultralytics (YOLO26) | 8.4.198 | AGPL-3.0 | detector + pose + classifier (ADR-003) |
| Ultralytics BYTETracker | (in ultralytics 8.4.198) | AGPL-3.0 | per-camera tracker (ADR-004; supervision 0.31 no longer ships ByteTrack) |
| prometheus-client | 0.26.0 | Apache-2.0 | metrics |
| nvidia-ml-py | 13.615.71 | BSD | GPU metrics |
| PostgreSQL | 17.11 | PostgreSQL | persistence (ADR-007) |
| MediaMTX | 1.21.3 | MIT | RTSP "NVR" simulator |
| Toxiproxy | 2.13.0 | MIT | network fault injection |
| Prometheus / Grafana | 3.15.1 / 13.2.5 | Apache-2.0 / AGPL-3.0 | monitoring |

## 3. Runtime topology (ADR-001, ADR-010)

```
 host (Windows, native)                                     docker compose
 ┌──────────────────────────────────────────────┐      ┌──────────────────────────┐
 │ API process  (uvicorn app.main:app :8000)     │      │ postgres :5432           │
 │  REST /api/v1, WS /ws/live, /metrics          │◄────►│ mediamtx :8554 (RTSP)    │
 │  WorkerSupervisor ─spawn─► Pipeline worker    │      │ toxiproxy :18501-18510   │
 │  OutboxDispatcher, Retention, LiveState       │      │ prometheus :9090         │
 │  serves frontend/dist (offline)               │      │ grafana :3000            │
 │                                               │      └──────────────────────────┘
 │ Pipeline worker process (:9101 /metrics)      │   scripts/demo/rtsp_simulator.py (host):
 │  CameraSource thread x N ─► LatestFrameSlot   │   ffmpeg -re -stream_loop -1 <STREAMS_DIR>/camNN.mp4
 │  InferenceScheduler thread (GPU)              │        -> rtsp://mediamtx:8554/camNN
 │   detect ─► track ─► associate ─► FSM ─► EventService ─► PostgreSQL / spool / evidence
 │  telemetry ─► mp.Queue ─► API                 │   app reads rtsp://localhost:185NN/camNN (via toxiproxy)
 └──────────────────────────────────────────────┘
```

Why a separate worker process: GPU OOM / native crash in decode or inference must not take down the API,
the dashboard or persisted state. The supervisor restarts it with backoff (spec 11.4, 11.5).

## 4. Module contracts (backend/app)

Contract files already written (do not change signatures without updating this doc):
`streaming/frame.py`, `inference/types.py`, `ppe/types.py`, `ppe/geometry.py`, `config/settings.py`,
`observability/metrics.py`, `workers/telemetry.py`, `camera/uri.py`, `core/redaction.py`.

### 4.1 camera/health.py — camera lifecycle FSM + backoff (FR-CAM-003/004/005)
```python
class CameraState(StrEnum): DISABLED, CONNECTING, ONLINE, DEGRADED, OFFLINE, RECOVERING, ERROR
class Backoff:  # bounded exponential with full jitter
    def __init__(self, base_s, max_s, rng: random.Random | None = None)
    def next_delay(self) -> float      # min(max, base*2**attempt) * rng.uniform(0.5, 1.0); attempt += 1
    def reset(self) -> None
    attempt: int
class CameraHealth:
    def __init__(self, camera_id, cfg: IngestConfig, nominal_fps: float | None = None, clock=time.monotonic, wall=time.time)
    state: CameraState; reason: str; since_wall: float
    def on_connecting(self) -> None          # CONNECTING first time, RECOVERING if it was ever ONLINE
    def on_connected(self, nominal_fps: float | None) -> None
    def on_frame(self) -> None               # updates last_frame, decoded fps window
    def on_error(self, reason: str, fatal: bool = False) -> None  # OFFLINE (or ERROR if fatal e.g. 401/404)
    def on_disabled(self) -> None
    def evaluate(self) -> CameraState        # called ~2 Hz by watchdog: frame age / fps => ONLINE/DEGRADED/OFFLINE
    def last_frame_age(self) -> float | None
    decoded_fps: float; reconnect_count: int; decode_errors: int
    online_seconds: float                    # accumulated ONLINE|DEGRADED time
    last_recovery_s: float | None            # OFFLINE/RECOVERING start -> first frame after reconnect
```
Rules: ONLINE requires a frame within `degraded_frame_age_s` and decoded fps >= `degraded_fps_ratio*nominal`
(fps check only after 3 s of data). Age in (`degraded_frame_age_s`, `stall_timeout_s`] => DEGRADED.
Age > `stall_timeout_s` => OFFLINE with reason "stalled: no frame for Xs" (the source must then reconnect).
Fatal errors (auth 401/403, 404 not found, invalid URI, missing credentials) => ERROR; still retried at max backoff.

### 4.2 camera/source.py — ingest threads (FR-CAM-002/005/006/007/009)
```python
class CameraSource(threading.Thread):  # one per camera, daemon
    def __init__(self, camera: CameraRuntimeConfig, ingest: IngestConfig, slot: LatestFrameSlot,
                 health: CameraHealth, convert_fps: float, fault_hooks: FaultHooks | None = None)
    def stop(self, timeout: float = 5) -> None  # sets flag, closes container, joins
```
`CameraRuntimeConfig` (dataclass in camera/source.py): id, stream_uri, credential_ref, transport, enabled, target_fps.
- rtsp(s): `av.open(resolve_uri(...), options={"rtsp_transport": transport}, timeout=(open_timeout_s, read_timeout_s))`.
- file://: deterministic demo source; loops the file and paces decode to the stream's frame rate (real-time).
- Decode all packets (`thread_type="AUTO"`), but call `to_ndarray(format="bgr24")` only at `convert_fps`
  (= convert_fps_factor * target_fps) to save CPU; put Frame into the slot.
- Any exception: `health.on_error(redact(e), fatal=...)`, close container (always, in `finally`), sleep
  `backoff.next_delay()` (interruptible by stop), reconnect. Backoff resets after `stable_reset_s` ONLINE.
- A watchdog thread (one for all cameras, in the worker) calls `health.evaluate()`; when it returns OFFLINE due
  to stall, it calls `source.request_reconnect()` which interrupts the blocked read (close container).
- Never log or expose the resolved URI; every error string goes through `redact()`.

### 4.3 streaming/slot.py — bounded latest-frame slot (NFR-PERF-006, spec 11.3)
```python
class LatestFrameSlot:
    def put(self, frame: Frame) -> bool   # returns True if an unconsumed frame was overwritten (count drop)
    def take(self) -> Frame | None        # returns and clears
    def peek_new(self) -> bool
    depth: int  (0 or 1); dropped: int
```
Thread-safe; notifies a shared `threading.Condition` passed at construction so the scheduler can wait.

### 4.4 streaming/scheduler.py — fair batch scheduler
```python
class InferenceScheduler:
    def __init__(self, slots: dict[str, LatestFrameSlot], target_fps: dict[str, float], max_batch: int,
                 max_batch_wait_ms: float, cond: threading.Condition, clock=time.monotonic)
    def next_batch(self, timeout: float) -> list[Frame]   # [] on timeout
    def set_target_fps(self, camera_id, fps); add_camera; remove_camera
```
A camera is due when its slot has a new frame and `now >= next_due[cam]`. Due cameras ordered by
`last_served` ascending (round-robin fairness); take up to `max_batch`. After serving:
`next_due = max(next_due + 1/fps, now - 1/fps)` (no burst catch-up after a stall).

### 4.5 inference — adapters (FR-DET-001/002/005/006/009)
- `inference/ultralytics_detector.py`: `UltralyticsCascadeDetector(models: ModelsConfig, models_dir)`
  implements `Detector`. Person stage: `models.person` (COCO detector or `*-pose` model; keep class 0,
  attach keypoints when the result has them). PPE stage: `models.ppe` with `ppe_class_map`; when
  `cascade_skip_ppe_without_person`, frames with no person skip the PPE model. Batched `predict(list, half,
  imgsz, conf, verbose=False)`. `stage_ms` filled per frame (batch time / batch size). `version` =
  `"<file stem>@<sha256[:12] of weights>"` for each stage, joined.
- `inference/mock_detector.py`: `MockDetector(script)` deterministic, used by tests and `backend: mock`.
  Script: per camera, a function of frame_id returning detections (e.g. person with/without helmet).
- `inference/classifier.py`: `UltralyticsHeadClassifier(spec, uncertain_below)` -> HELMET_PRESENT /
  HELMET_ABSENT / UNCERTAIN (top-1 prob < uncertain_below => UNCERTAIN).
- Model files: `models/` (git-ignored). `ModelSpec.weights` relative to models dir or an Ultralytics name.

### 4.6 tracking/tracker.py (FR-TRK-001/002/003)
```python
class PersonTracker(Protocol):
    def update(self, persons: list[Detection], frame: Frame) -> list[TrackedPerson]
    def reset(self) -> None
class ByteTrackAdapter:  # ultralytics.trackers.BYTETracker, one instance per camera - NEVER shared
    def __init__(self, cfg: TrackingConfig, frame_rate: float)
def make_tracker(cfg, frame_rate) -> PersonTracker
```
Keypoints travel through `sv.Detections.data["keypoints"]`. lost_track_buffer frames = lost_track_buffer_s * fps.

### 4.7 ppe/head_roi.py, ppe/association.py, ppe/zones.py (FR-DET-003/004/005/007)
```python
def head_roi(person: TrackedPerson, cfg: AssociationConfig, frame_w, frame_h) -> tuple[Box, str]  # (roi, "keypoints"|"geometry")
class Associator:
    def __init__(self, cfg: AssociationConfig, thresholds: RuleThresholds, classifier: HeadClassifier | None)
    def assess(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]
def zone_of(person_box, zones: list[ZoneRuntime], w, h) -> str | None      # foot point (bottom-centre)
def filter_ignored(persons, zones, w, h) -> list[TrackedPerson]            # drop persons in "ignore" zones
def apply_privacy_masks(image, zones) -> np.ndarray                        # blur "privacy" polygons (copy)
```
Association algorithm (deterministic, greedy, one helmet/head per person):
1. Visibility gate: person conf < person_conf, box height < min_person_height_px, or box top within 2 px of
   the frame top (head truncated) => UNCERTAIN ("not_assessable:<why>").
2. ROI: keypoints mode uses visible (conf >= keypoint_min_conf) nose/eyes/ears (>= min_head_keypoints);
   ROI centred horizontally on them, height = 1.6 * s, width = 1.4 * s, shifted up by 0.35 * s where
   s = max(spread of head keypoints, 0.45 * shoulder width, 0.18 * person height). Otherwise geometry:
   x1+margin*w .. x2-margin*w, y1-up*h .. y1+top*h. ROI head size < min_head_px => UNCERTAIN.
3. Candidate helmet h for person p: ioa(h, roi) >= min_ioa, centre-distance/diag(roi) <= max_center_dist,
   helmet width / person width within helmet_width_ratio. Score = ioa * conf * (1 - dist). Same for heads.
4. Greedy assignment over all (score desc) pairs so a helmet is never credited to two people (FR-DET-004).
5. Outcome: helmet (conf >= helmet_conf) and no head => HELMET; head (conf >= head_conf) and no helmet =>
   NO_HELMET; both and |Δconf| < conflict_margin => ambiguous; neither => ambiguous ("no_head_evidence").
   Ambiguous + classifier => classify ROI crop (HELMET_PRESENT->HELMET, ABSENT->NO_HELMET, else UNCERTAIN);
   without classifier => UNCERTAIN. Helmet on ground / in hand fails step 3 => no compliance credit.

### 4.8 ppe/state_machine.py (FR-TRK-004..008, spec 7)
```python
class TrackPPE:   # per (camera, track)
    state: PPEState; episode_seq: int; first_violation_at: float | None; last_seen: float
class PPEStateMachine:   # one per camera
    def __init__(self, camera_id: str, policy: TemporalPolicy, track_expiry_s: float)
    def update(self, observations: list[Observation], ts: float) -> list[Transition]
    def expire(self, ts: float) -> list[Transition]   # tracks unseen > track_expiry_s; CONFIRMED -> CLEARED("track_lost")
    def state_of(self, track_id) -> PPEState
    def set_policy(self, policy)                      # hot reload
```
Votes use the last `confirm_m` *decisive* observations (HELMET/NO_HELMET); UNCERTAIN observations are kept in
the window but never vote. Transitions:
- UNKNOWN/COMPLIANT/CLEARED -> SUSPECTED on `suspect_n` consecutive NO_HELMET (episode_seq += 1, first_violation_at = ts of first).
- any non-confirmed -> COMPLIANT on `compliant_n` HELMET in last `compliant_m` decisive.
- SUSPECTED -> CONFIRMED when count rule (`confirm_n` NO_HELMET in last `confirm_m` decisive) and/or duration
  rule (NO_HELMET evidence spanning >= min_duration_s with >= 80% of decisive votes NO_HELMET) per `mode`.
- SUSPECTED -> UNKNOWN when UNCERTAIN share of the last `confirm_m` observations >= unknown_ratio.
- CONFIRMED -> CLEARED on `clear_n` HELMET in last `clear_m` decisive (hysteresis) or track expiry.
- CONFIRMED stays CONFIRMED on UNCERTAIN / mixed evidence (no flapping, FR-TRK-006).
- CLEARED -> COMPLIANT / SUSPECTED per the rules above.

### 4.9 alerts/event_service.py, alerts/spool.py, evidence/store.py (FR-ALT-001..004/007/008/009)
- Episode id: `f"{run_id}:{camera_id}:{track_id}:{episode_seq}"`; idempotency key =
  `sha256(camera_id|track_id|zone_id|rule_id|episode_id)` hex. `run_id` = uuid4 of the worker start.
- On CONFIRMED: (a) cooldown: same camera+track has an event whose episode ended < cooldown_s ago => reopen it
  (status unchanged, `reconfirm_count += 1`, `episode_ended_at = NULL`), no new alert; (b) spatial dedupe: an
  event on the same camera+rule with last_seen within spatial_dedupe_s and IoU(last_box, box) >= spatial_dedupe_iou
  => merge (new track adopts the event), count `ppe_events_deduplicated_total{reason}`; (c) else create event +
  evidence + outbox rows in ONE transaction (`INSERT ... ON CONFLICT (idempotency_key) DO NOTHING`).
- While CONFIRMED: update `last_seen_at` / `last_box` at most every `last_seen_update_s`.
- On CLEARED: set `episode_ended_at`.
- DB failure: append the full event payload as one JSON line to `spool/events.jsonl` (flush + fsync), metric
  `ppe_events_spooled_total`; a replay thread retries every 2 s with the same idempotency key, then rewrites the
  spool file atomically (tmp + replace). Evidence is written before the DB insert so spooled events keep it.
- Evidence: `evidence/YYYY/MM/DD/<event_uuid>.jpg` (opaque id, no names), full frame with privacy masks applied,
  the violating person box + head ROI drawn. Before writing: `shutil.disk_usage` free < min_free_bytes => skip
  with `evidence_error="disk_low"`; OSError (ENOSPC etc.) => `evidence_error="write_failed"`. Never raises.

### 4.10 Database schema (PostgreSQL, Alembic `backend/alembic`) - ADR-007
| table | columns (pk first; * = indexed) |
|---|---|
| users | id uuid, username unique, password_hash (argon2id), role (admin/operator/viewer), enabled, created_at |
| sessions | token_hash (sha256 hex) pk, user_id fk, created_at, expires_at*, revoked_at |
| cameras | id text, name, stream_uri, credential_ref, transport, enabled, site, location, target_fps, created_at, updated_at |
| camera_health | id bigserial, camera_id fk*, state, reason, recorded_at* (state change history only) |
| zones | id text, camera_id fk*, name, kind (monitor/ignore/privacy), polygon jsonb, enabled |
| ppe_rules | id text, name, ppe_class, camera_id null, zone_id null, severity, enabled, thresholds jsonb, temporal_policy jsonb, updated_at |
| model_versions | id text (name@hash), model_name, version, artifact_hash, license, config jsonb, first_seen_at |
| events | id uuid, idempotency_key unique, camera_id fk*, track_id, zone_id, rule_id fk, episode_id, status* (OPEN/ACKNOWLEDGED/RESOLVED), severity, reason, confidence_summary jsonb, first_seen_at, confirmed_at*, frame_captured_at, persisted_at, last_seen_at, episode_ended_at, reconfirm_count, last_box jsonb, evidence_path, evidence_error, model_versions jsonb, feedback (TRUE_POSITIVE/FALSE_POSITIVE/null), feedback_note, acknowledged_by, acknowledged_at, resolved_by, resolved_at, dashboard_rendered_at, created_at, updated_at |
| alert_outbox | id bigserial, event_id fk, channel, status* (PENDING/DELIVERED/FAILED), attempts, next_attempt_at*, last_error, created_at, delivered_at; unique(event_id, channel) |
| event_actions | id bigserial, event_id fk*, actor_id, action (ACKNOWLEDGE/RESOLVE/FEEDBACK/REOPEN), note, created_at |
| audit_logs | id bigserial, actor_id, action, resource, created_at*, metadata jsonb (never secrets) |
| system_metrics_snapshots | id bigserial, ts*, metric_name, labels jsonb, value (benchmark rollups) |
| settings | key text pk, value jsonb, updated_at (evidence retention etc.) |

## 5. Workstreams

| WS | Scope | Depends on | Status |
|---|---|---|---|
| A | camera/health, ppe/head_roi, association, zones, state_machine + unit tests | contracts | see BUILD_STATUS |
| B | camera/source, streaming/slot, scheduler + tests (file + RTSP) | A (health) | |
| C | inference adapters, tracker, model training/eval scripts | contracts | |
| D | db, alembic, repositories, event_service, spool, evidence + integration tests | contracts | |
| E | FastAPI app, auth, routes, WS hub, supervisor, outbox dispatcher, retention | D | |
| F | pipeline worker wiring, telemetry, metrics, resource sampler | A-D | |
| G | React dashboard | API contract (docs/system-design/10-...) | |
| H | RTSP simulator + fault injection CLI, benchmark + soak scripts | compose | |

## 6. Test strategy (spec 19)

- `tests/unit`: pure logic, no I/O, < 5 s total. Run: `uv run pytest tests/unit -q`.
- `tests/integration` (`-m integration`): needs `docker compose up -d` (Postgres `ppe_test`, MediaMTX).
- `tests/fault_injection`: camera disconnect/stall/auth, DB outage, disk full, worker crash, outbox retry.
- `tests/performance` (`-m slow`): 1/4/10 camera load; results to `docs/benchmarks/`.
- Frontend: vitest unit tests + Playwright e2e against the running app.
