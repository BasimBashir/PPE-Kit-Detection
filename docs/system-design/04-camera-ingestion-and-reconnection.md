# 04 - Camera ingestion and reconnection

**Status:** VERIFIED (fault-injection tested) · **Related ADRs:** ADR-002 · **Modules:** `camera/source.py`, `camera/health.py`, `camera/uri.py`

## Purpose and scope
How streams are opened, health is judged and failures are recovered (FR-CAM-002..009).

## Assumptions
- RTSP over TCP by default; MediaMTX returns 404 when a camera is not publishing (treated as transient).

## Diagram
```mermaid
sequenceDiagram
  participant SRC as CameraSource
  participant H as CameraHealth
  participant WD as Watchdog (2 Hz)
  participant NVR as RTSP server
  SRC->>NVR: av.open(url, timeout=(5,5))
  NVR-->>SRC: frames
  SRC->>H: on_frame() → ONLINE
  Note over NVR: camera unplugged / stalled / network fault
  WD->>H: evaluate(): frame age > 1.5 s → DEGRADED
  WD->>H: evaluate(): age > 5 s or rate < 10% nominal → OFFLINE (stalled)
  WD->>SRC: request_reconnect()
  SRC-->>H: on_error(redacted reason) (404/EOF/timeout → OFFLINE; 401/403/creds → ERROR)
  loop bounded backoff with jitter (1, 2, 4, 8, 10 s cap)
    SRC->>NVR: reconnect
  end
  NVR-->>SRC: frames
  SRC->>H: on_frame() → ONLINE, last_recovery_s recorded
```
Source: [`diagrams/camera-reconnect-sequence.mmd`](diagrams/camera-reconnect-sequence.mmd)

## Component responsibilities
- `CameraSource`: open (5 s), read (5 s timeout), decode all packets, convert at 2× target FPS, reconnect with backoff.
- `CameraHealth`: lifecycle state (`DISABLED, CONNECTING, ONLINE, DEGRADED, OFFLINE, RECOVERING, ERROR`), decode FPS,
  uptime, reconnect count, last recovery time.
- `CameraWatchdog`: 2 Hz evaluation; frame age > 1.5 s → DEGRADED, > 5 s → OFFLINE; decode rate < 10 % of nominal →
  OFFLINE (a trickle of straggler frames is not "alive" - found by fault injection).

## Interfaces / contracts
`CameraRuntimeConfig`, `FaultHooks` (test/demo only). Errors classified: 401/403/credential/config → ERROR (retried at
max backoff); everything else (EOF, 404, timeouts, `InvalidDataError`) → OFFLINE with fast backoff.

## Data ownership
Health lives in memory; state changes are appended to `camera_health` (uptime reporting).

## Concurrency / process boundaries
One thread per camera; the watchdog never closes a container cross-thread (PyAV read timeouts break blocked reads).

## Failure modes
Measured detection/recovery per fault: [benchmarks/fault_injection.md](../benchmarks/fault_injection.md).

## Security / privacy
No credentials leave the worker: stream URIs are stored credential-free and resolved from `env:` references at connect time; every stream error passes `core.redaction.redact()`. PyAV error strings embed `user:pass@` - verified redacted in tests and logs.

## Performance implications
Decoding every packet costs CPU (~10 % of a core per 720p30 stream on the i5-12400F); NVDEC is a Phase-9 option.

## Alternatives considered
OpenCV VideoCapture; GStreamer; DeepStream (ADR-002).
