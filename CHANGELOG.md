# Changelog

## Unreleased

- Phase 4 verified live end to end: fall and fight rules confirmed **in the running app** on real test clips
  (OmniFall syn / SCFD test) through the RTSP demo path, evidence snapshots persisted - `docs/ml/phase4-log.md` §8.
  Fight alerts need `models.yaml fight_weights` (off by default, MVP weights).
- Per-camera enable/disable of rules pinned by an integration test (`backend/tests/integration/test_rule_scope.py`):
  a disabled rule of either check family never reaches the worker, and one state machine exists per (camera, rule).
- Demo tooling: annotated behaviour videos rendered from real test splits and synthetic poses, plus a standalone
  test-split scan CLI (`scripts/demo/behaviour_video.py`, `behaviour_synthetic_video.py`, `scan_test_segments.py`).

## 0.5.0 - 2026-11-24
Phase 5: attendance - a door-facing camera gets a two-point entry/exit line, crossings are matched to operator-enrolled
people, and a report shows minutes outside per day against a per-person budget. Decision record
[ADR-016](docs/decisions/ADR-016-attendance.md).

- Door zones are a third zone shape (2 normalised points, drawn in the walking direction of an entry, migration 0005);
  the crossing detector uses the tracked foot point with a deadband and a 5 s track-gap re-learn guard, and runs on the
  privacy-masked frame.
- Identity is deliberately **not** face recognition: a torso HSV histogram matched against an enrolled gallery, with
  state-constrained matching (EXIT only from someone inside, ENTER only from someone outside). Unmatched crossings stay
  in the Unknown bucket with their descriptor and crop so an operator can assign or enrol them later; every assignment
  rebuilds derived presence intervals, so corrections heal.
- Crossings ride the existing bounded `EventService` queue (droppable, idempotent sha256 key, breaker-aware metric
  `ppe_crossings_total`); `events` is unchanged, names live on persons/crossings only.
- API `/api/v1/attendance`: persons CRUD + gallery, crossings list/image/assign, and a day-clipped report (site time
  zone, 92-day cap, per-day outside/over-budget minutes and totals). New **Attendance** screen with crops and Unknown
  filter. Viewer reads, operator writes; zones stay admin.
- Cut and documented: no per-hour budget, no crossing spool on DB outage, no WS live updates, enrolment via crossing
  crops only. Identity accuracy is unmeasured on site footage - thresholds are defaults.
- Tests: backend 163 passed (new door-geometry, descriptor and day-clipping unit tests + attendance API integration),
  frontend lint/vitest/build green.

## 0.4.0 - 2026-11-23
Phase 4: behaviour and safety checks (fall, fight, phone use during work hours, stalling) as rule-driven checks, per
person, with per-rule shift schedules. Decision record [ADR-015](docs/decisions/ADR-015-behaviour-checks.md); measured
results and what failed in [docs/ml/phase4-log.md](docs/ml/phase4-log.md).

- Behaviours reuse the whole PPE path (ADR-015): `ppe_class` in {fall, fight, phone, stalling}, each with an assessor
  on the `Associator` interface, so the N-of-M state machine, dedupe, evidence, spool, outbox, retention and UI apply
  unchanged. ABSENT = the behaviour is happening; PRESENT = normal; UNCERTAIN = not assessable (never a violation).
- **Fall** = a fast sequence, not a posture: torso upright (<= 35 deg) -> lying (>= 60 deg) within 2 s with the hips
  dropping >= 0.25 standing body height, held down for the rule's duration policy. Slow lying down, bending and
  squatting never fire (unit-tested).
- **Fight** = a pair-keypoint GRU on CPU; training (`extract_fight_sequences.py`) and serving (`app/behaviour/fight.py`)
  share the same features: 2 x 17 keypoints, pair-centred and body-height normalised, 1.4 s window @ 5 Hz.
- **Phone use** = COCO class 67 over a 1.8x grown hand/head ROI (new COCO detection stage, `models.phone`), dropouts
  <= 2 s bridged. **Stalling** = keypoint stillness (5th-95th percentile range over 10 s): stillness, not intent -
  scope it to a work-area zone + schedule.
- Per-rule **shift schedule** (IANA tz + days + "HH:MM-HH:MM" windows; an overnight window belongs to its start day),
  editable in the UI (`ScheduleEditor`), seeded per preset, exposed on `Rule` (`schedule`, migration 0004). Off shift
  a rule emits no observations at all. Only new runtime dependency: `tzdata`.
- Measured, honestly low: fall OmniFall "in the wild" onset recall 17.4 % @ 5 fps / 27.6 % @ 10 fps; confirmable
  falls caught 0.65 % / 1.5 %; 1.09 false alarms/h. Fight GRU val F1 0.65 @ 0.55 (RLVS), cross-dataset F1 0.39
  (SCFD). Phone-stage COCO val2017 class 67 AP50 0.598 @ 640 / 0.547 @ 1280 (a close-up proxy). Site footage is
  required before any field accuracy claim; the fight weights are MVP-only.
- Cost: 10 x 4K @ 5 fps drops from 20.7 to 13.8 processed fps with four behaviour rules + the phone stage; the delta
  is association (22 ms p50 vs 0.7), dominated by the quadratic pair sequencer - vectorising it is the next step.
- Tests: backend 148 passed (behaviour unit tests + phone-stage integration), frontend lint/vitest/build green.

## 0.3.0 - 2026-11-20
Phase 3: multi-PPE (vest, gloves, goggles, safety shoes) with per-site mandatory/secondary rules, operator presets, configurable item catalogue, 4K support.

- Rules are item-generic (ADR-014): `ppe_rules.ppe_class` in {hard_hat, safety_vest, gloves, goggles, safety_shoes}, new `mandatory` flag (migration 0003) - mandatory rules alert, secondary rules are recorded and listed with a "secondary" tag but never push a live alert. The Phase 1 associator now matches any item to its body region from pose keypoints (head/eyes, torso, each wrist, each ankle); one bare hand or foot is a violation, non-visible regions stay UNCERTAIN (never a violation). Threshold keys renamed `present_conf`/`absent_conf`/`min_roi_px` (Phase 1 names still accepted; migration rewrites stored rules).
- Operator presets ("projects"): a named item -> mandatory/secondary map, created/edited/applied in Rules & settings (`GET/PUT/DELETE /api/v1/presets`, `POST /api/v1/presets/{name}/apply` with camera/zone scope, idempotent). construction/welding presets seeded on first start.
- Item catalogue is configuration (`models.yaml ppe_items`): detector classes, body region, labels, teacher question. New enforced class = detector training + config entry (no code); `GET /api/v1/ppe-items` feeds the UI. Second detector (`models.items`) covers the four new items; the measured helmet stack is unchanged. Teacher review asks one item-specific question per crop; asymmetric policy unchanged (never suppresses).
- 4K streams measured and supported (docs/benchmarks/phase3-4k.md): NVDEC hardware decode (`ingest.hwaccel: cuda`, software fallback), single resize shared by all model stages, preview pipeline shrunk before masking. 10 x 4K@25fps: host CPU 92% -> 57%; sizing tables for 10-50 cameras at the recommended 5 fps inference (one 24 GB GPU + ~16 cores per 10 cameras) and analysed at full 25 fps inference (marked hypothetical), plus a full-rate accuracy note.
- Network sharding (on/off): `PPE_WORKER_SHARD=i/N` splits cameras deterministically across N app instances sharing one PostgreSQL (off by default; docs/benchmarks/phase3-4k.md section 4).
- Tests: backend 112 passing incl. multi-item association (bare hand beats gloved hand, ground vest not credited, config-only new class end to end), secondary-rule no-outbox (SQL), preset API flow; frontend 16 passing; weights `ppe_items_yolo26m.pt` = plan C (mAP50 0.851 on the held-out Construction-PPE val; cross-dataset holdout on HB1204 shows a convention gap, site fine-tune expected; plans A/B/C compared in docs/ml/phase3-log.md).

## 0.2.0 - 2026-11-18
Phase 2: asynchronous teacher review, measured last-layer fusion, distillation flywheel.

- Teacher review layer (ADR-013): an external vision decision model re-checks head crops after the on-site decision
  — asynchronously, never on the alert path, off by default. Providers: Google Gemini Decisions (`gemini-decision-1`),
  the SystemOne wire protocol (self-hosted Jev/Clef), and a deterministic fake for tests. Asymmetric policy: the
  teacher may escalate a missed violation (new event via the normal path) or tag an alert LIKELY_FALSE_ALERT for
  human review; no code path can suppress or change an alert (unit test over the full probability grid + integration
  test asserting no UPDATE/DELETE). Reviews are stored (`teacher_reviews`, migration 0002) as a distillation dataset
  with an export script; `GET /api/v1/teacher/summary` + event-detail teacher row + Prometheus metrics.
- Benchmarks, honest negatives included: logistic regression beats the Laya decision model as the last layer
  (AUROC 0.9968 vs 0.9928, misses 4.7% → 3.2% at ≤2% FAR — docs/ml/fusion_eval); live Gemini teacher on 351 public
  CC0 crops (fused AUROC 0.972, 20/26 of our missed violations escalated, 0 suppressed — docs/ml/teacher_eval);
  distillation retraining runs end to end but shows no gain over continued training at 22–230 teacher labels
  (3 seeds with controls — docs/ml/distill_eval).
- Correctness: cooldown/dedupe state is re-seeded from the events table on worker restart, so a restart no longer
  re-alerts the same person (matching on person-box position, never restart-meaningless track ids; DB-down falls
  back to empty state).
- Performance (docs/benchmarks/phase2-performance.md): TensorRT FP16 measured (2.4× per-call at batch 1,
  1.25–1.35× at production batches), NVDEC probed and rejected (CPU decode faster on the dev host), CPU-only
  capacity measured (~5 cameras at 2 FPS), tracking placement measured (stays on the inference thread). Linux
  + systemd and UDP transport runbook sections added. 8 h soak clean: 50.2/50 FPS, 10/10 cameras online,
  0 worker restarts / 0 reconnects, worker RSS drift +0.9 MB/h (docs/benchmarks/soak_8h).
- Dashboard: admin Users page (create, role change, enable/disable, self-lockout guard); camera-source ingest loop
  under unit test; teacher timeout default 5 s → 30 s (live Gemini p95 ≈ 8 s truncated real calls).

## 0.1.1 - 2026-11-03
- Deployed model weights shipped via Git LFS; `start_demo.py` detects LFS pointer files and prints a `git lfs pull` hint.
- Fixed an uptime-dependent DB-outage test that failed on fresh CI runners.
- ADR-005…012 completed to the full template (trade-offs, consequences, evidence, follow-ups) with verified evidence;
  broken test/report references fixed across docs; README expanded with full setup and usage.

## 0.1.0 - 2026-11-02
First complete, measured portfolio release.

- 10-camera RTSP ingest (PyAV) with health state machine, stall watchdog, bounded backoff, latest-frame slots and a fair
  FPS-paced batch scheduler.
- YOLO26 cascade: pose person model + fine-tuned helmet/head detector (Hard Hat Workers, CC0) + optional head-crop
  classifier; per-camera ByteTrack; head-ROI association with global greedy matching.
- Temporal PPE state machine (N-of-M decisive votes, hysteresis, UNKNOWN for occlusion), idempotent events with
  cooldown and head-position spatial dedupe, evidence snapshots, durable outbox, DB-outage spool with circuit breaker.
- FastAPI REST + WebSocket, role-based auth, React dashboard (overview, camera wall with overlays, events review,
  configuration, rules, system health, demo control).
- Supervised worker process; Prometheus metrics + alert rules + Grafana dashboard.
- RTSP simulator with fault injection; one-command start (`scripts/start_demo.py`); benchmark, fault-injection and soak
  harnesses with generated reports.
