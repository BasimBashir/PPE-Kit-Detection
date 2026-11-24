# ADR-015: Phase 4 Behavioural Safety Analytics - behaviour and safety checks (fall, fight, phone use, stalling) as rule-driven checks

**Status:** ACCEPTED
**Date:** 2026-11-23

## Context / problem
The user's Phase 4 scope (2026-11-20): per person, detect a **fall** confirmed by its temporal sequence, **fighting**,
**phone use during work hours** and **stalling**, with "normal" = no event. Requested and **dropped**: "angry"
(emotion inference) and "not working" (undefined). Generic "accident" detection is parked until a client proposal.
The deployment region is the Gulf (UAE/KSA), so the output is an operator alert with evidence, never a judgement
about a person, and shift schedules matter (phone/stalling are only violations during work hours).

Phase 3's item rules already gave every observable a full path: rule -> associator -> N-of-M state machine ->
event + dedupe + evidence + spool + outbox + shift schedule -> UI. Phase 4 must reuse it, not fork it.

## Decision
1. **Behaviours are rule checks, not a second pipeline.** `Behaviour` in {fall, fight, phone, stalling} is a
   `ppe_class` in the existing rule table (migration 0004 adds `schedule`; `kind` on `/ppe-items`). Each rule gets an
   assessor with the `Associator` interface (`assess_deferred`/`take_pending`), so the state machine, dedupe,
   evidence, spool, outbox, retention and the operator UI apply unchanged. Outcome mapping: ABSENT = the behaviour
   is happening (violation evidence), PRESENT = normal, UNCERTAIN = not assessable (never a violation).
2. **Fall = a fast sequence, not a posture.** Torso upright (<= 35 deg from vertical) -> lying (>= 60 deg) within
   `fall_window_s` (2 s) while the hip centre drops >= `drop_frac` (0.25) of the standing body height, then held
   down for the rule's duration policy. Slow lying down, bending (hips stay up) and squatting (torso stays upright)
   never start an episode. No torso keypoints -> UNCERTAIN.
3. **Stalling = keypoint stillness only.** Every keypoint (and the box centre) stays within `idle_max_range`
   (0.08) body heights (5th-95th percentile over `idle_window_s` = 10 s). This measures stillness, **not intent**:
   scope it to a work-area zone + shift schedule. No public ground truth exists for stalling; only timer
   correctness is tested.
4. **Phone = COCO class 67 over a grown hand/head ROI.** The phone stage is a COCO detector (`models.phone`)
   run in the same batch as the other detectors; a phone box with IoA >= 0.3 over a 1.8x grown hand or head ROI is
   evidence; gaps shorter than `phone_gap_s` (2 s) are bridged so an intermittently detected phone still builds an
   episode. The rule's shift schedule gates enforcement.
5. **Fight = a small pair-keypoint GRU on CPU.** Two tracked people closer than `fight_pair_dist` (1.2) mean body
   heights form a pair; `pair_features` (2 x 17 keypoints, pair-centred, body-height normalised) is shared by
   training (`scripts/ml/extract_fight_sequences.py`) and serving (`app/behaviour/fight.py`), so they cannot drift.
   Sequences are resampled to the checkpoint's rate; the window is **1.4 s (SEQ_LEN 8 @ 5 Hz)** - user-approved
   because SCFD clips are ~2 s long. One person alone is PRESENT (no fight); a nearby partner without a full
   sequence is UNCERTAIN. `target_fps` must be >= 5.
6. **Per-rule shift schedule.** `Schedule` = IANA `tz` + days (0 = Monday) + windows ("HH:MM-HH:MM"; an overnight
   window belongs to the day it starts; end exclusive). Off shift: no observations, so episodes close and tracks
   expire. Editable per rule in the UI (`ScheduleEditor`), seeded per preset.
7. **No new runtime dependencies except `tzdata`**; no paid APIs; all thresholds are operator-editable rule
   settings or documented engineering constants (`BehaviourConfig`).

## Alternatives considered
- **"Angry" emotion detection** - rejected: emotion inference from CCTV faces is unreliable (AffectNet-class
  models top out around 59-61 % accuracy on close-up faces, far worse at CCTV distance) and the user dropped it.
- **"Not working"** - rejected: no definable observable.
- **A separate behaviour pipeline** (own state machine, events, alerts) - rejected: duplicates the event path,
  dedupe, evidence and UI for no gain.
- **RGB video transformers (RWF-2000 class)** for fights - rejected for now: GPU cost on the inference path, and no
  measured need against the pair-keypoint GRU; revisit if site data shows the GRU is insufficient.
- **Track stitching / re-identification** - deferred: fall recall is limited by pose quality and cross-dataset
  domain shift, not only by ID switches; look at it again with site footage.
- **Teacher review of behaviours** - rejected: the teacher question rubric is PPE-only; adding behaviour questions
  needs labelled data first.

## Trade-offs
- Reusing the PPE state machine means behaviour events inherit PPE-style confirmation windows; a fall therefore
  needs the rule's duration policy before it alerts (seconds, not instantly).
- The fight model needs `target_fps >= 5`; lower rates leave pairs "warming_up" (UNCERTAIN), never false alarms.
- Behaviour assessment runs on the inference worker thread; in crowded scenes the pair sequencer is quadratic in
  the number of close people (measured below).

## Consequences
- UNCERTAIN without pose: if the person keypoint model cannot see a tracked person, behaviours never fire.
- Stalling reports stillness; a worker thinking at a desk is "still". Zone + schedule scoping is mandatory in
  deployment, not optional.
- Fall/fight/phone accuracy is measured on public datasets only (below); **site footage is required** before any
  customer-facing accuracy claim.
- The fight weights are MVP weights: they must be retrained on licensed or client data before commercial use.
- 10-camera 4K throughput drops from 20.7 to 13.8 processed fps with four behaviour rules + the phone stage
  (`docs/benchmarks/results/p4_uhd_cams10_fps5_nvdec.json`).

## Evidence / benchmark
- Fall (`docs/ml/fall_eval_*.json`, OmniFall): wild ItW test set, onset recall 17.4 % @ 5 fps / 27.6 % @ 10 fps;
  falls confirmable while lying: 0.65 % / 1.5 %; false alarms 1.09/h. Synthetic 20.3 % onset recall @ 5 fps.
  **Low cross-domain recall - a site-data problem, stated plainly in `docs/ml/phase4-log.md`.**
- Fight: RLVS val clip F1 0.65 @ 0.55; SCFD cross-dataset F1 0.39 @ 0.5 (`docs/ml/fight_eval.json`).
- Phone: COCO val2017 class 67 AP50 0.598 @ 640 / 0.547 @ 1280 (`docs/ml/phone_coco_eval.json`) - a proxy only,
  COCO phones are close-ups.
- GPU/CPU cost: `docs/benchmarks/results/p3_uhd_cams10_fps5_nvdec.json` vs `p4_uhd_cams10_fps5_nvdec.json`.

## Follow-up actions
- Collect site footage (with a licence) for fall/fight/phone and re-measure; retrain the fight GRU on it.
- Vectorise `PairSequencer._sequence` (currently ~0.2 ms per pair, 16 ms/frame at 20 close people) without
  changing features, and re-measure.
- Add behaviour questions to the teacher rubric once labelled data exists.
