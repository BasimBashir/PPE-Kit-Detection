# models/

The three deployed weights are committed with **Git LFS** (AGPL-3.0, see docs/model-and-data-licenses.md). Clone with
Git LFS installed, or run `git lfs install && git lfs pull` afterwards; `scripts/start_demo.py` refuses to start on LFS
pointer files. To reproduce them instead:

| File | How to obtain |
|---|---|
| `yolo26s-pose.pt` | Ultralytics asset (COCO person + keypoints), downloaded automatically when missing |
| `ppe_yolo26s.pt` | `python scripts/dataset/prepare_hardhat_dataset.py --out <work>/hardhat` then `python scripts/dataset/train_ppe_detector.py --data <work>/hardhat/yolo/data.yaml` |
| `head_cls_yolo26n.pt` | `python scripts/dataset/train_head_classifier.py --data <work>/hardhat/cls` |

Model provenance (file hash, training run, metrics) is recorded in `docs/ml/` and in the `model_versions` table at runtime.
For a GPU-less smoke test set `backend: mock` in `configs/models.yaml`.
