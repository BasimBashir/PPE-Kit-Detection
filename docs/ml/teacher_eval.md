# Teacher review: live evaluation of Gemini Decisions (`gemini-decision-1`)

Raw numbers: [`teacher_eval.json`](teacher_eval.json). Script: `scripts/ml/teacher_benchmark.py`. Responses are cached
(`~/ppe-work/teacher/cache.jsonl`), so a rerun makes no requests.

## What was sent

- **Only public CC0 images**: head crops from the hard-hat dataset's test split (oracle boxes). No site footage was sent.
- The **free tier allows Google to use submitted data**. That is why site footage must never go to the free-tier
  endpoint, and why `teacher.enabled` is `false` by default.
- Set: 150 helmet + 150 no-helmet crops (seed 42), plus **all 65 crops our classifier gets wrong** on the full test split
  (2,378 crops); 351 after overlap. Because the set is enriched with errors, the counts below are **not deployment rates**.
- Requests: 351 benchmark calls (all HTTP 200, no 429 at a 1.5 s pace) plus 1 manual endpoint probe. About 1.2k input
  tokens per crop. Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/decisions` with an `x-goog-api-key`
  header and one `noul` question.

## Results (report half, n = 168; weight and thresholds chosen on the other half)

The two halves are split by file-name hash. Weight and thresholds were tuned on the tune half (n = 183) under a
budget: at most 2% of true alerts wrongly tagged and at most 2% of compliant samples falsely escalated. The tuned
values were **w = 0.25, escalate > 0.20, false-alert < 0.61**.

| P(no hard hat) | Accuracy | AUROC | ECE |
|---|---|---|---|
| Teacher alone | 0.750 | 0.946 | 0.174 |
| Our classifier alone | 0.821 | 0.966 | 0.132 |
| Fused (w = 0.25) | 0.845 | 0.972 | 0.109 |

| Asymmetric policy on the report half | Count |
|---|---|
| Violations our model missed → **ESCALATED** | **20 of 26** |
| Our false alerts → tagged LIKELY_FALSE_ALERT | 2 of 4 |
| True alerts wrongly tagged LIKELY_FALSE_ALERT | 1 of 59 |
| Truly compliant samples falsely escalated | 3 of 79 (above the 2% tuning budget) |
| Alerts suppressed, resolved or changed | **0, by construction** (see `test_teacher_can_never_suppress_an_alert`) |

Teacher latency is p50 1.40 s and p95 7.88 s (n = 351). This is fine for a background reviewer and is the reason the
teacher is never on the alert path.

## Honest reading

- On its own the teacher is **worse** than our classifier on these crops. Its value is as a second, differently-trained
  opinion: fusing them recovers most of our misses at a small false-escalation cost.
- Oracle crops are cleaner than runtime head ROIs, so the tuned thresholds are **not** adopted as defaults. Re-tune
  them on reviewed site data before enabling the teacher.
- For distillation, confident teacher labels are clean but scarce. With |p − 0.5| ≥ 0.4 there were 22 labels at 100%
  precision; with ≥ 0.3 there were 120 at 100%; with ≥ 0.2 there were 230 at 90.9%.
  `scripts/ml/export_teacher_labels.py` defaults to a margin of 0.4. Operator feedback overrides teacher labels.
