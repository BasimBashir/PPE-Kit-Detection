# Distillation flywheel: retraining the head classifier on teacher labels

Raw numbers: [`distill_eval.json`](distill_eval.json) (keys `margin_0.4`, `margin_0.2`). Script:
`scripts/ml/distill_finetune.py` (`--selftest` covers the split, the label rule and the threshold search). Every
experiment has a **control**: the identical fine-tune without any teacher crops, which isolates what the labels
contribute from what mere continued training contributes.

## Setup (both experiments)

- Teacher labels from the 351 cached reviews (`teacher_eval.md`) via `export_teacher_labels.py --source cache`; a
  crop keeps its label when |p_teacher − 0.5| ≥ margin. Split by file-name hash: 70% fine-tune, 30% holdout (never
  trained on).
- Fine-tune from `models/head_cls_yolo26n.pt` with the `train_head_classifier.py` recipe (30 epochs, imgsz 96,
  batch 256, patience 10), one run per seed; best.pt selected on the original val split. `models/` is unchanged;
  weights stay in `~/ppe-work/distill`.
- The alarm threshold is tuned on the fine-tune crops' **teacher** labels at ≤ 2% false alarms; holdout predictions
  are scored against the crops' ground truth. `rest_of_test` = every public test crop except that margin's fine-tune
  crops, so counts are **not comparable across margins** (margin 0.2 excludes more crops, including most hard ones).

## Iteration 1 — margin 0.4, seed 42 (22 labels, 100% precision)

Labels: 22 used (17 helmet, 5 no_helmet), 329 skipped; fine-tune 15 (12/3), holdout 7 (5/2). Training 5.6 min.

| Model | Holdout acc. (n = 7) | Holdout ECE | Miss @ FA thr. | Rest-of-test missed violations (of 573) |
|---|---|---|---|---|
| Shipped | 0.857 (6/7) | 0.105 | 0/2 | 52 |
| Control (no teacher crops) | 1.000 | 0.054 | 0/2 | 43 |
| Distilled (+15 teacher crops) | 1.000 | 0.062 | 0/2 | 28 |

The flywheel runs end to end, but the 6/7 → 7/7 holdout gain also appears in the control, and the labels are 0.08% of
the training data: one seed proves the mechanics, not the labels.

## Iteration 2 — margin 0.2, seeds 42/43/44 (230 labels, 90.9% precision)

Labels: 230 used (53 helmet, 177 no_helmet), 121 skipped; **14 of the 166 fine-tune labels are wrong** against ground
truth. Holdout n = 64 (18 helmet, 46 no_helmet; 39 violations); rest-of-test n = 2,212 (459 violations). Training
2.7–3.0 min per run.

| Seed | Control missed | Distilled missed | Control holdout acc. | Distilled holdout acc. | Control holdout ECE | Distilled holdout ECE |
|---|---|---|---|---|---|---|
| 42 | 25 | 23 | 0.875 | 0.859 | 0.130 | 0.139 |
| 43 | 27 | 25 | 0.859 | 0.859 | 0.112 | 0.124 |
| 44 | 21 | 27 | 0.875 | 0.844 | 0.131 | 0.130 |
| mean (min–max) | 24.3 (21–27) | 25.0 (23–27) | 0.870 (0.859–0.875) | 0.854 (0.844–0.859) | 0.124 (0.112–0.131) | 0.131 (0.124–0.139) |

**Verdict: the distillation effect is not demonstrated at this label volume.** Control and distilled ranges overlap on
every metric; the control is nominally ahead on holdout accuracy and missed violations, and the distilled model wins
only on rest-of-test ECE (0.0079 vs 0.0095). All six retrained models do beat the shipped model on the holdout
(shipped: 0.797 acc, 0.153 ECE) — that gain comes from continued training, not from the teacher. With 14 mislabelled
training crops, teacher noise appears to offset the signal; profitable distillation likely needs operator-confirmed
labels rather than raw confident teacher output.

## Caveats

- (a) Labels are teacher-generated, not ground truth: 100% precision at margin 0.4 (22/22) but 90.9% at margin 0.2
  (209/230), and label errors concentrate exactly where the model is weakest.
- (b) The holdout is 30% of the same public test split the teacher saw. It measures the flywheel mechanics, not site
  performance. At n = 64 one crop is 1.6 accuracy points.
- (c) A real deployment would accumulate labels from reviewed site crops over time, with operator decisions
  overriding teacher labels (`export_teacher_labels.py --source db`).
