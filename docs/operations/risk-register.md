# Risk register

Owner: project lead. Reviewed: 2026-11-02. Status values: OPEN, MITIGATED (control implemented + tested), ACCEPTED.

| # | Risk | Impact | Likelihood | Mitigation / control | Verification | Status |
|---|---|---|---|---|---|---|
| R1 | Helmet near a person (ground, hand, neighbour) credited as worn | False compliance (missed violation) | Medium | Head ROI from keypoints, IoA + centre-distance + width gates, global greedy matching (ADR-005) | tests/unit/test_association.py hard negatives | OPEN |
| R2 | Occluded / tiny / truncated head classified as "no helmet" | False alert | High | `UNCERTAIN` outcome never votes; visibility gate; classifier only for ambiguous cases | unit tests; demo FP review | OPEN |
| R3 | Ten streams exceed hardware capacity | Latency, dropped frames | Medium | Latest-frame slots, fair batching, cascade skip, 2/5/10 FPS profiles | docs/benchmarks | OPEN |
| R4 | RTSP instability | Camera downtime unnoticed | High | Per-camera FSM, frame-age watchdog, backoff+jitter, metrics + alert rules | fault_injection.md: unplug/stall/timeout/refused/NVR restart all recovered | MITIGATED |
| R5 | Unbounded queues | Memory growth, stale alerts | Low | 1-frame slots, bounded telemetry queue, frame-age-at-inference histogram | soak_1h.md RSS/VRAM slope; slots depth ≤ 1 | MITIGATED |
| R6 | Tracker ID switches | Duplicate events for one person | Medium | Spatial dedupe + cooldown (ADR-006) | event counts per camera | OPEN |
| R7 | Model/data licence incompatibility | Release blocked | Low | AGPL repo; licence inventory (docs/model-and-data-licenses.md) | inventory reviewed | MITIGATED |
| R8 | Dataset leakage | Inflated metrics | Medium | Near-duplicate hash grouping before split | dataset_report.json | MITIGATED (source lacks video ids - residual risk) |
| R9 | Disk fills with evidence | Pipeline degradation | Low | Free-space guard, retention job, disk metric + alert | fault_injection.md disk_full_60s + unit tests | MITIGATED |
| R10 | GPU OOM / worker crash | Inference outage | Low | Separate worker process, supervisor restart, OOM handling | fault_injection.md worker crash 0.6 s / 14.7 s | MITIGATED |
| R11 | Database outage | Lost events | Low | fsync JSONL spool + idempotent replay; DB-down banner | fault_injection.md DB outage: 2 spooled, 0 lost | MITIGATED |
| R12 | Demo footage unrepresentative (stock clips, construction) | Misleading claims | High | "SIMULATED SOURCES" banner, docs state domain gap, no accuracy claims on demo | README / benchmark report wording | MITIGATED |
| R13 | Credentials leak via FFmpeg error messages | Security | High (observed) | `redact()` on all stream errors + logging filter; credential_ref only in API | tests/unit/test_redaction_and_uri.py | MITIGATED |
| R14 | Domain gap: dataset is 416 px close-ups, demo is 720p wide shots | Missed small heads | High | Measure on demo clips, min-size gate -> UNCERTAIN; document | demo evaluation | OPEN |
| R16 | Non-hard-hat headwear (caps, head wraps) accepted as helmet | Missed violations (false compliance) | High (observed on demo cam06) | Hard-negative data collection + fine-tune; per-scene recall evaluation; operator review | docs/benchmarks/README.md §6 | OPEN |
| R15 | Single dev host is a desktop RTX 3090 | Edge performance unknown | High | Results labelled with hardware; edge checklist | docs/benchmarks | ACCEPTED |
