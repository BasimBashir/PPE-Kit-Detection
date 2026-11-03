# models/

Weights are **not committed** (size + AGPL/derivative licensing, see docs/model-and-data-licenses.md). Expected files:

| File | How to obtain |
|---|---|
| `yolo26s-pose.pt` | downloaded automatically by the app / `scripts/bootstrap/download_models.py` (Ultralytics asset) |
| `ppe_yolo26s.pt` | `python scripts/dataset/prepare_hardhat_dataset.py --out <work>/hardhat` then `python scripts/dataset/train_ppe_detector.py --data <work>/hardhat/yolo/data.yaml` |
| `head_cls_yolo26n.pt` | `python scripts/dataset/train_head_classifier.py --data <work>/hardhat/cls` |

Model provenance (file hash, training run, metrics) is recorded in `docs/ml/` and in the `model_versions` table at runtime.
For a GPU-less smoke test set `backend: mock` in `configs/models.yaml`.
