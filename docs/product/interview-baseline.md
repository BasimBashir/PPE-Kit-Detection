# Interview baseline → engineering corrections

Spec section 2 requires keeping the interview-derived baseline distinct from the enhancements. This page maps every design
element proposed in the original whiteboard discussion (a generic on-ship helmet-detection design exercise) to what this
project implements and **how it is verified**. It is an independent portfolio project; no company code, data or footage is used.

## Problem as posed
Offline system, ~10 cameras on a ship, RTSP from a local NVR, detect people without helmets, raise local alerts, edge hardware
(Jetson-class or mini-PC). Targets discussed: ~80 % recall, ≤ 1 false positive per 2 hours, ~5,000+ annotated images.

## Mapping

| # | Whiteboard proposal | Assessment | What this project does | Evidence |
|---|---|---|---|---|
| 1 | One YOLO model, classes person + helmet | Sound baseline | Kept as a configurable baseline profile (`person.weights: yolo26s.pt`, `head_roi: geometry`, classifier off) and benchmarked against the full cascade | docs/benchmarks (ablation) |
| 2 | Associate helmet to the upper 10–15 % of the person box via IoU / distance | Incomplete: fails for helmets carried at chest height or near a neighbour | Head ROI from pose keypoints (fallback: top-of-box geometry); IoA + centre distance + plausible width + confidence; **global** greedy matching so one helmet is never credited to two people; temporal confirmation | ADR-005, tests/unit/test_association.py |
| 3 | YOLO-Pose "assigns IDs" | **Incorrect** — pose models estimate boxes + keypoints, they do not track | Pose model for boxes/keypoints only; **ByteTrack** (separate component) per camera; IDs never shared across cameras; no cross-camera Re-ID (not required) | ADR-004, tests/unit/test_mock_detector_tracker.py |
| 4 | YOLO-OBB for helmets | Not justified: helmets are not strongly oriented objects | Axis-aligned detector; OBB only if measured tilt failures justify it (none measured) | ADR-003 |
| 5 | Head-region classifier | Reasonable | Optional `yolo26n-cls` head-crop classifier with HELMET_PRESENT / ABSENT / **UNCERTAIN**, run only for ambiguous cases (cascade) | ADR-005, docs/ml/head_classifier_eval.json |
| 6 | Alert after 10–15 continuous frames | Arbitrary: duration depends on FPS | N-of-M **decisive** votes (UNCERTAIN never votes), optional duration mode, hysteresis; API reports `effective_confirm_seconds`; benchmarked at 2/5/10 processed FPS | ADR-006, docs/benchmarks |
| 7 | Occluded head ⇒ no helmet | Wrong default | Explicit `UNKNOWN` / `UNCERTAIN` — insufficient visibility never produces an alert | tests/unit/test_state_machine.py |
| 8 | 5,000 images ⇒ 80 % recall | Unsupported: image count is not a performance guarantee | CC0 dataset (5,000 images) with near-duplicate-grouped split; test-split metrics reported separately from system-level behaviour on demo streams; no claim the 80 % / FP target is met | docs/ml, docs/testing |
| 9 | "1 FP per 2 hours" | Ambiguous for 10 cameras | Reported both **per camera-hour** and **per system-hour** from operator-reviewed events | /api/v1/evaluation/report |
| 10 | Freeze weights + add layers | Oversimplified | Systematic fine-tuning of a pre-trained detector; frozen-backbone vs full fine-tune compared by measurement | docs/ml/ppe_detector_eval*.json |
| 11 | Jetson / Orange Pi "is sufficient" | Unverified | Capacity measured (decode, inference, end-to-end) on the available host with 10 RTSP streams; edge device claims explicitly deferred to on-device benchmarks | docs/system-design/15-capacity-planning-and-performance.md |
| 12 | (not discussed) camera failures, backpressure, durability | Gap | Per-camera health FSM, stall watchdog, backoff+jitter, latest-frame queues, durable outbox, DB-outage spool, worker supervision, metrics + alert rules | fault-injection report |
