# Runbook: Model update and rollback

## Update
1. Train/evaluate: `scripts/dataset/train_ppe_detector.py` / `train_head_classifier.py` → metrics in `docs/ml/`.
2. Copy the new weights into `models/` under a new file name; point `configs/models.yaml` at it.
3. Restart the worker (System health → Restart worker). The new `<name>@<sha12>` version appears in `/ready` and on
   every new event (`model_versions`), and is recorded in the `model_versions` table.

## Rollback
Point `configs/models.yaml` back to the previous file and restart the worker. Old weights are never overwritten in place.

## Verify
`/api/v1/system` → `models` shows the intended versions; compare event rate / FP feedback before and after.
