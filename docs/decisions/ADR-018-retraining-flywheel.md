# ADR-018: Retraining flywheel — teacher-mined FP/FN, developer-approved batches, gated weight promotion

**Status:** ACCEPTED (classify adapter live; detect / sequence adapters pending)  
**Date:** 2026-11-25

## Context / problem
Sites drift: new PPE colours, uniforms, seasons, lighting, camera moves. ADR-013 already stores every asynchronous
teacher (VLM) review next to the model's own score, but nothing turned the disagreements into a better model. The
distillation study showed teacher labels alone, at low volume, do not reliably beat continued training. So any
retraining must be human-approved and must prove itself before it replaces deployed weights.

## Decision
One check-agnostic loop, `scripts/ml/flywheel.py` (full design: [docs/ml/flywheel.md](../ml/flywheel.md)):

1. **Mine**: a review is a mistake when confident truth disagrees with the model (`p_model ≥ 0.5`). Truth is the
   operator's feedback if any, else the teacher when `|p_teacher − 0.5| ≥ margin`. FN = missed violation,
   FP = false alarm. Frames are filed under the correct label per check and deduped by review id.
2. **Threshold → notify**: when a check's inbox reaches its configured threshold, it becomes a review batch with a
   contact sheet, and the developer is notified (console, `notifications.jsonl`, `FLYWHEEL_WEBHOOK_URL`).
3. **Approve / reject**: the developer looks once. Moving a file relabels it, deleting it drops it. Nothing is trained
   without approval.
4. **Benchmark**: the developer signs a frozen benchmark set (hash-checked) plus per-metric tolerances once per check.
5. **Train**: the champion is fine-tuned on its original train set + 70% of approved mined frames (hash split).
6. **Gate**: promote only if the candidate is no worse than the champion on every signed benchmark metric (within
   tolerance) **and** makes ≥ `mined_min_gain` fewer errors on the held-out 30% of mined frames.
7. **Promote**: archive the old weights by hash, `os.replace` atomically, append `models/registry.jsonl`. The app
   loads the new weights on restart.

`teacher_reviews` gains `ppe_class` (migration 0007) so mined mistakes sort per check. Trainer adapters by check:
`classify` (hard-hat head classifier, live), `detect` (PPE-items detector, phone), `sequence` (fight GRU), `rule`
(fall, stalling, outside-time: no weights; mined misses inform threshold tuning).

## Alternatives considered
- **Fully autonomous promotion (no human):** rejected. Teacher labels are pre-labels (ADR-013 measured false
  escalations), and in a safety system a silent regression in missed violations is the worst failure.
- **Retraining inside the API process / on a timer regardless of data:** rejected. GPU contention with live
  inference, and retraining without new approved data only adds risk.
- **Promote on "better on the mined set" alone:** rejected. It overfits to the mined failures; the frozen benchmark is
  the regression guard.
- **A training service + DB tables + UI now:** deferred. A CLI with file state is enough for one site, auditable, and
  schedulable. The state files map one-to-one to tables if a UI is needed later.

## Trade-offs
Promotion takes effect on app restart (no hot reload). Small batches give a statistically thin mined hold-out, so the
benchmark tolerance is the real guard. Detector and behaviour checks are mined and approved but not yet retrained:
detectors need boxes on full frames, behaviours need clip-level teacher input.

## Evidence
Real run on 351 cached Gemini Decisions answers ([docs/ml/flywheel.md §8](../ml/flywheel.md)): 12 mistakes mined
(10 FN, 2 FP), 12/12 correct against ground truth. The candidate cut benchmark misses 43 → 36 but raised false
alarms 10 → 15, so the zero-tolerance gate **rejected** it. With a developer-signed +5 false-alarm tolerance it was
**promoted** (archive + atomic swap + registry), then rolled back. Training is deterministic (identical metrics on
re-run). Tests: `flywheel.py selftest` (mining truth table, gate, threshold → batch → approval on a temp dir).
Backend suite green.

## Follow-up actions
`detect` adapter (persist full frame + box per review); clip teacher for behaviours; per-camera caps on batch share;
shadow-run stage before promotion; classifier hot reload on registry change.
