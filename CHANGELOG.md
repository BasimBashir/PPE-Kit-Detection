# Changelog

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
