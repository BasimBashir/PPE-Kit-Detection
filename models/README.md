# models/

The deployed weights are committed with **Git LFS** (AGPL-3.0, see docs/model-and-data-licenses.md). Clone with
Git LFS installed, or run `git lfs install && git lfs pull` afterwards; `scripts/start_demo.py` refuses to start on LFS
pointer files. To reproduce them instead:

| File | How to obtain |
|---|---|
| `yolo26s-pose.pt` | Ultralytics asset (COCO person + keypoints), downloaded automatically when missing |
| `ppe_yolo26s.pt` | `python scripts/dataset/prepare_hardhat_dataset.py --out <work>/hardhat` then `python scripts/dataset/train_ppe_detector.py --data <work>/hardhat/yolo/data.yaml` |
| `head_cls_yolo26n.pt` | `python scripts/dataset/train_head_classifier.py --data <work>/hardhat/cls` |
| `yolo26s.pt` | Ultralytics asset (COCO; supplies "cell phone" = class 67 for the phone stage), downloaded automatically when missing |
| `fight_gru.pt` | `python scripts/ml/extract_fight_sequences.py --root data/fight/rlvs --pos Violence --neg NonViolence --out data/fight/rlvs.npz` then `python scripts/ml/train_fight.py train --data data/fight/rlvs.npz --out models/fight_gru.pt --report docs/ml/fight_eval.json` |

Model provenance (file hash, training run, metrics) is recorded in `docs/ml/` and in the `model_versions` table at runtime.
For a GPU-less smoke test set `backend: mock` in `configs/models.yaml`.
