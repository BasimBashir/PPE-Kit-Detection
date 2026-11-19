# Changelog

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
