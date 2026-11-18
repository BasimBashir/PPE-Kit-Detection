# Last-layer fusion benchmark: classifier vs logistic regression vs Laya

Question: does the text-only decision model Laya (`convaiinnovations/laya`, `typed-decisions` checkpoint,
ModernBERT-large 421M, PyPI `laya` 0.4.35, Apache-2.0) help as the final fusion step over the numeric signals
our vision models already produce? Machine-readable results: [`fusion_eval.json`](fusion_eval.json).

## Setup

- Data: CC0 hard-hat dataset, every labelled helmet / bare-head box at least 12 px wide. Val: 2,488 heads (514 violations),
  used for all fitting, prompt choice and thresholds. Test: 2,337 heads (571 violations), used once. Train was not
  used (the head classifier was trained on train crops).
- Features (`scripts/ml/fusion_features.py`, production settings, fp16, RTX 3090): head classifier P(no_helmet) on
  the 25%-expanded crop; `ppe_yolo26s` best-IoU (>= 0.3) helmet and bare-head confidences (imgsz 640, conf 0.25);
  box geometry. Boxes come from the labels, not from the detector (oracle localisation).
- Methods (`scripts/ml/fusion_benchmark.py`): classifier alone; standardized logistic regression on 8 numeric
  features; Laya zero-shot (`noul` probability); Laya + Platt scaling on val. For Laya, two serialisations (compact
  JSON, one-sentence text) x two questions were scored on val, and the most informative one was kept: the text state
  with "Is the worker missing a hard hat?".
- Metrics on test: accuracy at the val-best threshold; AUROC with a 1,000-resample bootstrap 95% CI (paired with
  logistic for the difference); miss rate on violations at the threshold that gives <= 2% false alarms on val; ECE
  (10 bins); fusion-step latency per decision. Vision-model time is excluded, because every method shares it.

## Results (test)

| Method | Accuracy | AUROC [95% CI] | AUROC minus logistic [95% CI] | Miss rate @ 2% FAR (test FAR) | ECE | Fusion latency / decision |
|---|---|---|---|---|---|---|
| Classifier alone | 0.973 | 0.9940 [0.9906, 0.9966] | [-0.0045, -0.0014] | 4.7% (1.9%) | 0.022 | ~0 (threshold) |
| **Logistic fusion** | **0.986** | **0.9968** [0.9947, 0.9983] | - | **3.2%** (1.7%) | **0.013** | 0.21 ms CPU |
| Laya zero-shot | 0.974 | 0.9928 [0.9879, 0.9963] | [-0.0075, -0.0016] | 4.2% (1.8%) | 0.282 | 35 ms GPU (batch 1), 1.8 ms GPU (batch 128) |
| Laya + Platt | 0.974 | 0.9928 [0.9879, 0.9963] | [-0.0075, -0.0016] | 4.2% (1.8%) | 0.018 | same as zero-shot |

Laya fine-tuning was not run. The package has no fine-tuning API for these checkpoints. Its documented route is a
separate RLCD training notebook, and `Agent.fit` applies only to the separate vegaml reader.

## Conclusion

On this task, Laya does not justify its cost. The plain logistic regression beats it on every metric. The AUROC
gap is significant (the paired CI excludes 0). Logistic also misses fewer violations at the same false-alarm budget
(3.2% vs 4.2%), and each decision is about 170x cheaper on a CPU than Laya's single-decision GPU call. Zero-shot Laya
is roughly equal to the classifier it reads. Its probabilities are badly over-confident (ECE 0.28) until Platt
scaling fixes them. It is also fragile to how the state is written: the same numbers as compact JSON scored a val
AUROC of 0.45 and 0.08 (inverted), against 0.98 to 0.998 as one sentence. Logistic fusion is the right last layer
here: it gives a measured gain over the classifier alone (accuracy 0.973 -> 0.986, miss rate 4.7% -> 3.2%).

Caveat: Laya is text-only. It cannot see the image, so it can only re-weigh signals the vision models already
produced. A numeric model fitted to those same signals is the natural ceiling it has to beat. Laya might add value
where the state includes free text, such as operator notes, site rules or shift context, but this benchmark does
not test that.
