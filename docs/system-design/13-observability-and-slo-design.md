# 13 - Observability and SLO design

**Status:** VERIFIED · **Related ADRs:** ADR-012 · **Modules:** `observability/*`, `deployments/docker/alert-rules.yml`, Grafana dashboard

## Purpose and scope
Metrics, alert rules and the targets they protect (spec 10).

## Assumptions
- Prometheus scrapes both processes every 5 s.

## Diagram

## Component responsibilities
`observability/metrics.py` is the single list of metric names (per camera, inference, events, resources).
`alert-rules.yml`: CameraOffline, StaleFrames, LowProcessedFps, HighFrameAgeAtInference, WorkerDown, GpuMemoryHigh,
DiskFreeLow, DatabaseUnavailable, OutboxBacklog, InferenceErrors - validated with `promtool`.

## Interfaces / contracts
`/metrics` (API), `:9101/metrics` (worker), `/api/v1/metrics` JSON summary for the dashboard.

## Data ownership
Prometheus local TSDB; dashboard history ring buffer (10 min) in the API.

## Concurrency / process boundaries
Each process exports what it owns; labels are bounded (camera_id, stage, class, channel, state).

## Failure modes
Windows has no prometheus process collector → our own psutil/NVML gauges.

## Security / privacy
No ids or secrets in labels.

## Performance implications
Targets (to validate, spec 5): alert P95 < 2 s after confirmation condition; UI < 1 s after persistence; queue depth ≤ 1.
Measured values in [15](15-capacity-planning-and-performance.md).

## Alternatives considered
OpenTelemetry tracing (deferred).
