# Caps fix: teacher-labeled fine-tuning on the cam06 cap crops (measured)

The demo's known failure: on the cam06 metal-workshop clip, workers wearing caps are classified compliant
(P(no helmet) low). Fix attempt measured here: extract head crops from that clip exactly like production
(pose-model head ROI with the same gates), teacher-label them with Gemini Decisions (the app's `noul`
question; stock Pexels footage — allowed by the data rule), retrain the head classifier with those labels
added, and measure the confidence shift on held-out crops. Pipeline: `scripts/ml/caps_extract.py` +
`scripts/ml/caps_finetune.py`; raw numbers in `caps_eval.json`.

## Setup

- **Crops**: 205-frame clip sampled at 3 fps, production head-ROI extraction (40 keypoints-mode, 15
  geometry-fallback), histogram-deduped to **55 distinct crops of 6 worker identities**.
- **Teacher (gemini-decision-1)**: 55/55 answered, 0 refusals/blocks. Mean p(no helmet) **0.793**. With the
  flywheel's margin-0.2 rule: **41 no_helmet, 0 helmet, 14 skipped as unconfident**. The teacher confirms
  the caps are "not wearing a hard hat" — the demo failure is the classifier, not the labeling concept.
- **Split by identity** (greedy-IoU tracks, never by frame): 5 identities / 41 crops train, **1 identity /
  14 crops holdout** — held-out crops come from a worker never seen in training.
- **Models** (identical recipe, seed 42, 30 epochs, imgsz 96): shipped `models/head_cls_yolo26n.pt`
  (untouched); control = CC0 train crops + CC0 teacher labels (reused from the distill study, sha-verified);
  distilled = control **+ the 41 caps teacher-labeled train crops**. Weights in `C:/Users/mianx/ppe-work/caps/`.

## Results

Held-out cap crops (14, never trained on) — the fix metric:

| model | mean P(no helmet) | share classified no_helmet at 0.5 |
|---|---|---|
| shipped | 0.044 | 0.0% |
| control (CC0 only) | 0.020 | 0.0% |
| distilled (+ caps labels) | **0.293** | **28.6%** |

CC0 hash-holdout regression check (64 crops, ground truth from the original test-split folders; method
identical to the distill study — shipped/control reproduce its recorded numbers exactly):

| model | accuracy | missed violations at 0.5 | false alarms at 0.5 | AUROC | ECE |
|---|---|---|---|---|---|
| shipped | 0.797 | 10 | 3 | 0.928 | 0.153 |
| control | 0.875 | 6 | 2 | 0.941 | 0.130 |
| distilled | **0.875** | **6** | **2** | **0.965** | **0.118** |

## Verdict

The teacher-labeled caps fine-tune moves the classifier decisively in the right direction: mean P(no helmet)
on never-trained cap crops rises 6.7x (0.044 -> 0.293) and 28.6% of them now cross the alert threshold,
while the control run (retraining without caps) does nothing. On CC0 it costs nothing: accuracy, missed
violations and false alarms at 0.5 are identical to control, and AUROC/ECE improve. What it does **not** do:
it does not flip the majority of held-out cap crops past 0.5 (71% still classified compliant), and the
<=2%-false-alarm threshold calibration degrades (training-set helmet-crop probabilities saturate toward 0,
so the study's threshold-from-teacher-helmet-probs rule yields a degenerate threshold; at the fixed 0.5
operating point nothing regresses). This is a measured, partial fix: report the confidence shift honestly,
do not claim cam06 becomes compliant end-to-end with this head classifier alone.

## Caveats

- **Same-clip holdout is not a new-site test.** The held-out worker shares camera, lighting and scene with
  training crops; only worker identity is held out. A different site could behave worse.
- **The holdout is small**: 1 worker, 14 crops (the 6.8 s clip contains only 6 identities; 80/20 by identity
  gives 5/1).
- **Teacher labels are not ground truth**: no human-labeled caps set exists here; all headline numbers use
  teacher confidence, only the CC0 check uses ground truth.
- **Stock Pexels footage**, not real site footage; single seed (42); single margin (0.2).
