# ADR-013: asynchronous teacher review (Gemini Decisions / SystemOne) + logistic last-layer fusion, distillation flywheel

**Status:** ACCEPTED  
**Date:** 2026-11-18

## Context / problem
The head classifier still misses hard negatives (caps, hoods, headwraps, bare hair in shadow) and a false alert
rate that is acceptable at 2% can still erode operator trust. A stronger vision model could catch some of these —
but putting a second model on the alert path would couple alert latency to its availability, and paid cloud
inference is out of budget. Privacy bounds the cloud option: free-tier terms allow the provider to use submitted
data, so real site footage must never leave the site.

## Decision
A **teacher** reviews head crops **asynchronously, after the on-site decision** (never on the alert path; off by
default). Review queue: every confirmed violation plus a rate-limited sample of tracks judged COMPLIANT — where
dangerous misses hide. Providers behind one protocol (`app/inference/teacher.py`): Google **Gemini Decisions**
(`gemini-decision-1`, one `noul` question returns P(no helmet); crops are base64 in the request; key only ever in
the `x-goog-api-key` header, registered for log redaction), **SystemOne** wire protocol (self-hosted Jev /
Cloudflare Clef; coded and unit-tested against a fake transport, unverified live), and a deterministic FakeTeacher
for tests.

Fusion and policy are **asymmetric** (`decide()`): fused = w·p_teacher + (1−w)·p_model. The teacher may ESCALATE a
compliant sample above `escalate_threshold` (a new event through the normal event + outbox path, reason "raised by
teacher review") or tag an alert LIKELY_FALSE_ALERT below `false_alert_threshold` (a review row only). No code
path deletes, suppresses, resolves or edits an alert — proven by a unit test over the full probability/threshold
grid and an integration test that captures SQL and asserts the event row and its outbox delivery are untouched.

Every review is stored (`teacher_reviews`, migration 0002: crop path, p_model, p_teacher, fused, verdict, latency)
as a **distillation dataset**; `scripts/ml/export_teacher_labels.py` exports confident labels (operator feedback
overrides teacher labels) and `scripts/ml/distill_finetune.py` retrains the head classifier on them with a
no-teacher control, evaluated on a hash-split holdout. **Logistic regression on the pipeline's numeric signals is
the last-layer fusion model** (measured, see below) — used by the benchmark, not (yet) on the runtime path: the
runtime fusion is the weighted mean above, which is the same family with w in place of a fitted coefficient vector.

## Alternatives considered
- **Laya (text-only typed-decision model) as the last layer**: measured on 2,337 labelled heads and lost to
  logistic regression on every metric (AUROC 0.9928 vs 0.9968, paired CI excludes 0; miss rate 4.2% vs 3.2% at the
  same false-alarm budget; ~170x the per-decision cost), while logistic fusion measurably improves on the
  classifier alone (4.7% → 3.2% misses). Rejected; kept as a documented negative result.
- **Laya as the image teacher**: impossible — it never sees pixels.
- **Cloudflare Clef as the image teacher**: the 27B model needs ~54 GB VRAM (too big for the deployment GPU);
  `clef-flash` 9B remains a drop-in option via the SystemOne adapter.
- **Teacher on the alert path / auto-resolving tagged alerts**: rejected — outage coupling and suppression risk.

## Trade-offs
Cloud teacher sees crops (public CC0 data only in the benchmark; `teacher.enabled: false` by default so nothing
leaves a site by accident). Teacher latency (p50 1.4 s, p95 7.9 s) is irrelevant off the alert path. On oracle
crops the tuned policy falsely escalated 3/79 compliant samples (above the 2% tuning budget), so the tuned
thresholds are deliberately **not** shipped as defaults — re-tune on reviewed site data before enabling. Only
person-box geometry is persisted for restart dedupe (ADR-006), not head ROIs.

## Consequences
New table + migration (`0002_teacher_reviews`), httpx moved to runtime deps, `GEMINI_API_KEY` in Settings
(SecretStr, redacted), `teacher:` config section, `GET /api/v1/teacher/summary`, teacher row in the event detail
view, review/failure/latency metrics. DB-down-at-worker-start still leaves a small duplicate-alert window
(ADR-006). Distillation at this label volume does **not** yet beat continued training (multi-seed, control vs
distilled overlapping) — profitable distillation needs operator-confirmed site labels, which the export path
already prefers.

## Evidence / benchmark
- Fusion: docs/ml/fusion_eval.{json,md} — logistic beats Laya (test, n=2,337, paired bootstrap).
- Teacher (live, 351 public CC0 crops, cached): docs/ml/teacher_eval.{json,md} — teacher alone AUROC 0.946,
  fused 0.972; 20/26 of our missed violations escalated; 0 alerts suppressed; false escalations 3/79.
- Distillation: docs/ml/distill_eval.{json,md} — margin 0.4 (22 labels) and margin 0.2 (230 labels, 90.9%
  precision), 3 seeds with controls: effect not demonstrated at this label volume; mechanics run end to end.
- Tests: backend/tests/unit/test_teacher.py (incl. `test_teacher_can_never_suppress_an_alert` over the full grid),
  backend/tests/integration/test_teacher_review.py (incl. `test_false_alert_tag_never_changes_the_event` — no
  UPDATE/DELETE issued), suite 98 passing.

## Follow-up actions
Re-tune w/thresholds on reviewed site data before enabling the teacher; accumulate operator-confirmed labels and
re-run the distillation study at real volume; verify the SystemOne adapter live (self-hosted Clef-flash); TLS
termination guidance (ADR-012).
