# BUILD_STATUS

Single source of truth for what is implemented and **verified**. `VERIFIED` = exercised in the running application and
covered by passing tests or a recorded measurement. Statuses: `TODO` · `IN_PROGRESS` · `IMPLEMENTED` · `VERIFIED` ·
`BLOCKED` · `NOT DONE`.

Spec: [PPE_Intelligence_Master_Spec.md](PPE_Intelligence_Master_Spec.md) (full scope, user decision 2026-11-02).
Licence: AGPL-3.0-or-later (user decision). Interview mapping: [docs/product/interview-baseline.md](docs/product/interview-baseline.md).

## Session 2026-11-02 - state at end of session
- Git: published 2026-11-03 at https://github.com/BasimBashir/PPE-Kit-Detection (`main`).
- Running: Docker services, simulator, application (for the soak test).
- Environment: Windows 11, i5-12400F, RTX 3090, venv outside OneDrive (`UV_PROJECT_ENVIRONMENT`).

## Phases
| Phase | Status | Evidence |
|---|---|---|
| 0 Discovery and design | VERIFIED | implementation-plan, ADR-000..012, system-design 00-16 + 10 diagrams, risk register, licence inventory |
| 1 Foundation | VERIFIED | uv project, config schema, Alembic (`alembic check` clean), JSON logs with redaction, health/ready |
| 2 Video pipeline | VERIFIED | GPU cascade test on real frames; file:// source implemented (not used by the demo) |
| 3 Single RTSP camera | VERIFIED | live fault campaign (unplug/stall/timeout/refused/creds/decoder) |
| 4 Ten cameras | VERIFIED | 10 × 5 FPS = 50.2 FPS processed, frame age p95 0.126 s (benchmarks) |
| 5 PPE detection + association | VERIFIED (unit + live) / accuracy gap OPEN | detector test-split metrics; association hard-negative unit tests; observed false compliance on caps (R16) |
| 6 Temporal FSM + alerts | VERIFIED | unit + integration tests; live events with evidence; dedupe/cooldown/spool |
| 7 Dashboard + workflows | VERIFIED | headless-Edge screenshots of all 8 pages; acknowledge/resolve/feedback via API tests; live WS |
| 8 Resilience + observability | VERIFIED | 13 fault scenarios; Prometheus rules (promtool), Grafana dashboard provisioned |
| 9 Optimisation + edge | PARTIAL | classifier batching +38 % saturation throughput; TensorRT/NVDEC/DeepStream/edge NOT DONE |
| 10 Portfolio release | IN_PROGRESS | README, demo script, test report, model/data cards; demo video + public repo pending (owner) |

## Feature checklist (spec FR ids)
| Area | Status | Notes |
|---|---|---|
| FR-CAM-001..009 | VERIFIED | UDP selectable but not tested (FR-CAM-002 TCP verified) |
| FR-DET-001..009 | VERIFIED | pose ROI (005) + head classifier (006) in cascade; UNKNOWN on occlusion (007) |
| FR-TRK-001..008 | VERIFIED | ByteTrack per camera (Ultralytics BYTETracker; supervision dropped it) |
| FR-ALT-001..009 | VERIFIED | sound channel (005) implemented, off by default, not exercised; clips (003 optional) NOT DONE |
| FR-UI-001..006 | VERIFIED | + Demo control page (stream count 0-10, per-camera and system faults) |
| FR-API-001..004 | VERIFIED | OpenAPI at /api/docs |
| FR-OFF-001..004 | VERIFIED | weights/assets local after first fetch; dashboard served by API; no runtime cloud call |
| Users/roles/auth | VERIFIED | argon2id, opaque tokens, 3 roles, rate limit; no users UI (API only) |

## Measured results (all generated; see docs/benchmarks/README.md)
| Measurement | Result | Source |
|---|---|---|
| Throughput 10 × 5 FPS | 50.2 / 50 FPS, frame age p95 0.126 s, GPU 37 %, VRAM 2.7 GB, RSS 2.8 GB | results.md |
| Throughput 10 × 2 / 10 × 10 | 20.1 / 20; 81.5 / 100 (not sustained) | results.md |
| Saturation (10 × 30 requested) | 72.6 FPS (52.5 before classifier batching) | results.md, results_before_classifier_batching/ |
| Detector test split | helmet mAP50 0.964, head mAP50 0.937 | docs/ml/ppe_yolo26s_eval.json |
| Frozen vs full fine-tune | no gain from freezing (0.964 vs 0.964 mAP50; mAP50-95 0.631 vs 0.640) | docs/ml/*freeze10* |
| Head classifier | top-1 0.973 (9.2 % bare heads → "helmet") | docs/ml/head_classifier_eval.json |
| Event → dashboard (NFR-PERF-005) | p50 24 ms, p95 52 ms, max 240 ms over 195 events | ui_latency.md |
| Time to confirm | ≥ 2.2 s by policy at 5 FPS; observed 2.1-3.6 s | events, ADR-006 |
| Fault recovery | 13/13 scenarios recovered; e.g. unplug 0.6 s / 7.8 s, worker crash 0.6 s / 14.7 s, DB outage 0 lost | fault_injection.md |
| 1 h soak | see soak_1h.md | soak_1h.md |
| Tests | backend 77 passed (unit + integration + GPU); frontend 14 passed; ruff + tsc clean | docs/testing/test-report.md |

## Known defects / limitations
- **False compliance on caps/head wraps** (R16) and **no alerts on distant small workers** (cam03) on demo footage.
- No labelled site footage → no event recall / FP-per-hour figures; interview targets not claimed.
- Alert counts inflated by looping clips (same workers re-enter as new tracks every 7-33 s).
- Cooldown/dedupe memory lost on worker restart; constant network latency undetectable without source timestamps.
- CPU-only profile, UDP transport, Linux/edge deployment: not verified. No Playwright E2E test suite (headless
  screenshots + WS regression test only).
- The worker's RTSP behaviour is tested live (fault/load campaigns), not by a dedicated pytest.

## Next exact tasks
1. Owner: reset demo data and record the demo video (weights are now in the repo via Git LFS).
2. Reset demo data before recording (`docker compose down -v`, delete `evidence/` and `spool/`), then follow
   docs/demo/demo-script.md.
3. Hard-negative data (caps, hoods, helmets in hand/on ground) + fine-tune; labelled clips for event-level recall.
4. TensorRT export + NVDEC decode; 8 h soak; on-device edge benchmark.
