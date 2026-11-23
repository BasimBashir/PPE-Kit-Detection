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
| 9 Optimisation + edge | PARTIAL | classifier batching +38 % saturation throughput; TensorRT FP16 measured (2.4× per-call at batch 1, 1.25-1.35× at batch 10, engines not wired into the pipeline) and NVDEC probed (CPU decode faster on dev host, not integrated) — docs/benchmarks/phase2-performance.md; DeepStream/on-device edge NOT DONE |
| 10 Portfolio release | IN_PROGRESS | README, demo script, test report, model/data cards; Phase 2 video (whole-team audience) pending |
| 11 Phase 2: teacher review + fusion + distillation | VERIFIED | ADR-013; docs/ml/fusion_eval + teacher_eval (live) + distill_eval; smoke-tested end to end (fake: all 4 verdicts + escalation visible; gemini: ~80 clean reviews); 0 alerts suppressed (grid + SQL-capture tests) |
| 12 Phase 3: multi-PPE rules + presets | PARTIAL | ADR-014; five items with mandatory/secondary rules, operator presets (API + UI), config-only item catalogue; network sharding `PPE_WORKER_SHARD` (unit-tested, off by default); items detector = plan C trained on Construction-PPE + HB1204, mAP50 0.851 on the unseen CPPE val (cross-dataset transfer gap flagged); 4K measured + sizing docs incl. hypothetical 25 fps full-rate table (docs/benchmarks/phase3-4k.md); NVDEC wired; live smoke VERIFIED (presets applied via API -> 5 rules hot-reloaded; 10x 4K dual-detector events with evidence + per-item overlay; gloves correctly UNKNOWN on wide clips, no false alerts) |

| 13 Phase 4: behaviour checks (fall, fight, phone, stalling) | PARTIAL | ADR-015; behaviours are `ppe_class` rule checks on the existing state machine ("angry"/"not working" dropped by the user, generic accident parked); fall = fast transition + hold (unit-tested: slow lying/bending/squat never fire), fight = pair-keypoint GRU on CPU (val F1 0.65, cross-dataset 0.39 - MVP weights), phone = COCO class 67 over grown hand/head ROI, stalling = keypoint stillness (no ground truth, timer tests only); per-rule shift `schedule` (migration 0004, UI ScheduleEditor); 4K benchmark `p4_uhd_cams10_fps5_nvdec` = 13.8 processed fps (pair sequencer is the bottleneck, follow-up); fall/fight/phone accuracy measured on public datasets only - **no field claim**, site footage required |

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
| Users/roles/auth | VERIFIED | argon2id, opaque tokens, 3 roles, rate limit; admin Users page (`config/users`: create, role, enable/disable, self-lockout guard) |

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
| 8 h soak | 50.2 / 50 FPS, 10/10 online, 0 restarts / 0 reconnects, worker RSS +0.9 MB/h | soak_8h.md |
| Last-layer fusion (test, n=2,337) | logistic AUROC 0.9968 beats Laya 0.9928; misses 4.7 % → 3.2 % at ≤2 % FAR | docs/ml/fusion_eval.md |
| Teacher review (live, 351 public crops) | fused AUROC 0.972; 20/26 missed violations escalated; 0 suppressed; p95 7.9 s | docs/ml/teacher_eval.md |
| Distillation (22/230 teacher labels, 3 seeds) | no measured gain vs control at this label volume | docs/ml/distill_eval.md |
| TensorRT FP16 (measured) | 2.4× per-call at batch 1; 1.25-1.35× at production batches; NVDEC probed, not worth it | docs/benchmarks/phase2-performance.md |
| Tests | backend 98 passed (unit + integration + GPU); frontend 16 passed; ruff + tsc clean | docs/testing/test-report.md |

## Known defects / limitations
- **False compliance on caps/head wraps** (R16) and **no alerts on distant small workers** (cam03) on demo footage; the asynchronous teacher-review layer (ADR-013, off by default) is the designed mitigation — 20/26 of our missed violations were escalated in the live benchmark.
- Distillation from raw confident teacher labels showed no gain over continued training at 22-230 public labels (controls overlap, 3 seeds); profitable retraining needs operator-confirmed site labels.
- No labelled site footage → no event recall / FP-per-hour figures; interview targets not claimed.
- Alert counts inflated by looping clips (same workers re-enter as new tracks every 7-33 s).
- Cooldown/dedupe memory is re-seeded from the `events` table on worker restart (person-box IoU only; empty state if
  the DB is down at start, so a post-crash duplicate is then still possible); constant network latency undetectable
  without source timestamps.
- CPU inference cascade measured (~5 cameras at 2 FPS on the dev host i5-12400F, inference only —
  docs/benchmarks/phase2-performance.md); full-app CPU profile, a live RTSP-over-UDP source, Linux/edge deployment:
  not verified (UDP transport is code-verified and documented). No Playwright E2E test suite (headless
  screenshots + WS regression test only).
- The worker's RTSP behaviour is tested live (fault/load campaigns); the ingest loop itself is unit-tested on local files (`test_camera_source.py`: flow, backoff/offline/recovery, watchdog), not against a live RTSP source.

## Next exact tasks
1. Record a Phase 3 clip: preset apply in the UI, multi-item overlay, secondary event on the Events page.
2. Record the Phase 2 video for the whole ALTAVE team (6-8 min, TTS + silent captioned; business-first).
3. Re-tune teacher w/thresholds on reviewed site data before enabling; accumulate operator-confirmed labels and re-run the distillation study at real volume.
4. Wire TensorRT engines into the pipeline (export path measured; the person-pose stage is the half that could lift the saturation ceiling); on-device edge benchmark.
5. Playwright E2E suite; live UDP source exercise.
