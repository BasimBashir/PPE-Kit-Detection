# ADR-014: Phase 3 Multi-PPE Compliance & Site Policy Templates - vest, gloves, goggles, safety shoes on the Phase 1/2 architecture

**Status:** ACCEPTED  
**Date:** 2026-11-20

## Context / problem
Sites differ in what they require: a construction site requires hard hat, vest and safety shoes; a welding bay
requires goggles and gloves. Operators must set, per site and per working zone, which items are **mandatory**
(alert), which are **secondary** (record only), and which are off. Phase 1 (on-site cascade: pose -> helmet/head
detector -> head ROI association -> N-of-M state machine -> event/outbox) and Phase 2 (asynchronous teacher review,
asymmetric: escalate or tag, never suppress) are measured and must not regress.

## Decision
1. **One rule = one item in one scope.** `ppe_rules.ppe_class` in {hard_hat, safety_vest, gloves, goggles,
   safety_shoes}; scope = camera (or all) + monitor zone (or full frame); `enabled`; new `mandatory` flag
   (migration 0003). Mandatory -> event + outbox (dashboard push, optional sound). Secondary -> the same event row,
   tagged `secondary`, **no outbox rows** (listed on the Events page, no live alert).
2a. **Operator presets ("projects").** A preset is a named item -> mandatory | secondary map (items left out =
   off), created and edited by an admin in Rules & settings (`PUT/DELETE /api/v1/presets/{name}`), stored in the
   `settings` table (no new table). Applying a preset to a scope (all cameras / one camera / one zone,
   `POST /api/v1/presets/{name}/apply`) creates or updates one rule per item - idempotent, so re-applying an edited
   preset updates the same rules and keeps their tuned thresholds. `configs/presets/*.yaml` (construction,
   welding) are seeded as editable presets on first start. No Project entity: one deployment serves one site.
2b. **Item catalogue is configuration, so new classes need no code.** `models.yaml ppe_items` (merged over the
   five built-ins) defines each item: detector present/absent class names, body region (head, eyes, torso, hands,
   feet), size gate, label, alert text and teacher question. Rules, presets, the UI (`GET /api/v1/ppe-items`),
   association and the teacher all read it. Adding e.g. ear muffs = train/extend the items detector with
   `earmuffs`/`no_earmuffs`, map them in `items_class_map`, add one `ppe_items` entry (test:
   `test_new_class_is_config_only`). A rule whose item is later removed is skipped with an error log, never a
   worker crash. A new body region (e.g. "neck") is the one case that needs code (`region_rois`).
2. **Same pipeline per item, different body region.** The Phase 1 associator is generalised, not duplicated:
   item detection = PRESENT evidence, bare-region detection (`no_vest`, `no_gloves`, `no_goggles`, `no_boots`;
   `head` for hard hats) = ABSENT evidence, matched by the same IoA / centre-distance / size gates and global greedy
   matching to the item's region from pose keypoints: head (hard hat), eyes/head (goggles), shoulders+hips (vest),
   each wrist (gloves), each ankle (shoes). Two-slot items: **any bare hand/foot = ABSENT** (one bare hand is a
   glove violation). No keypoints for hands -> UNCERTAIN (never a violation). Feet cut by the frame edge ->
   UNCERTAIN. The N-of-M state machine, dedupe (head ROI as identity cue), evidence, spool and outbox are unchanged.
3. **Helmet stack untouched; a second detector for the new items.** `ppe_yolo26s` (helmet/head, measured
   mAP50 0.964) and the head classifier keep their role. A separate items detector (`models.items`) detects the
   other four items + their "no_*" classes, run in the same batch as the helmet stage. Training data, model size,
   input size and the measured alternatives: [docs/ml/phase3-log.md](../ml/phase3-log.md).
4. **Teacher review per item.** One yes/no question per item on the region crop (`teacher.py QUESTIONS`); fusion
   and the asymmetric policy are unchanged, so the teacher still can never suppress an alert.
5. **Generic names where the meaning changed.** `Outcome.PRESENT/ABSENT` (was HELMET/NO_HELMET); thresholds
   `present_conf/absent_conf/min_roi_px` (Phase 1 names still accepted on input; migration 0003 rewrites stored
   rules).

## Alternatives considered
- **One combined detector for all five items** (retrain helmet together with the rest): the public multi-PPE data
  labels hard hats on far fewer images than Hard Hat Workers (5,000 images), so the measured helmet stack would
  be traded for an unmeasured one; keeping it costs one extra GPU call per batch, which production can afford.
- **A Project table / per-project rule sets in the DB**: one deployment serves one site today; presets + per-zone
  rules cover "construction vs welding area" without a new entity. Revisit for multi-site central servers.
- **Presets as files only** (rule-set YAMLs copied to rules.yaml): operators cannot edit them without server
  access; the settings-table presets keep the same simplicity but are editable in the UI and audited.
- **Item present = compliant, nothing seen = violation** (no "no_*" classes): absence of a detection is not
  evidence (occlusion, small objects) - Phase 1 already rejected this for helmets (ADR-005).

## Trade-offs
Gloves and goggles are small: at 720p a hand on a distant worker is ~10 px, below `min_roi_px`, so the rule
stays UNCERTAIN rather than alerting. The answers are higher-resolution streams (measured: Phase 3 log, 4K) and
close-angle cameras for zones where gloves/goggles are mandatory. Secondary events do not appear as live alerts
by design.

## Consequences
Migration 0003; `RuleSeed.mandatory`; `ModelsConfig.items`; overlay fields `ppe_class/item_roi/present_box/
absent_box`; rule create form in Rules & settings; Events show item + "secondary". Tests:
`backend/tests/unit/test_multi_ppe.py`.

## Evidence / benchmark
docs/ml/phase3-log.md (datasets, plans tried, per-class test metrics, what failed and why),
docs/benchmarks/results/p3_*.json (multi-item and 4K pipeline throughput).
