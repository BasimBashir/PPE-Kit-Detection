# Model and data licence inventory (spec 13.2 release gate)

Checked 2026-11-02; Phase 3 additions 2026-11-20. Code licence, weight licence and training-data licence are assessed separately.

| Asset | Used for | Source | Licence | Redistributed in this repo? | Notes |
|---|---|---|---|---|---|
| Ultralytics library 8.4.198 | training + inference | pypi.org/project/ultralytics | AGPL-3.0 (Enterprise licence available) | No (dependency) | Repo is AGPL-3.0-or-later to stay compatible |
| `yolo26s-pose.pt` | person+keypoints | github.com/ultralytics/assets releases | AGPL-3.0 (Ultralytics) | Yes - `models/`, Git LFS | COCO pre-training |
| `yolo26s.pt`, `yolo26n-cls.pt` | PPE fine-tune base; classifier base | github.com/ultralytics/assets releases | AGPL-3.0 (Ultralytics) | No - downloaded when training | COCO / ImageNet pre-training |
| `ppe_yolo26s.pt` (ours) | helmet/head detector | trained by `scripts/dataset/train_ppe_detector.py` | AGPL-3.0 (derivative of Ultralytics weights) + CC0 data | Yes - `models/`, Git LFS; reproducible | |
| `head_cls_yolo26n.pt` (ours) | head-crop classifier | `scripts/dataset/train_head_classifier.py` | AGPL-3.0 + CC0 data | Yes - `models/`, Git LFS; reproducible | |
| `ppe_items_yolo26m.pt` (ours, Phase 3) | vest/gloves/goggles/boots + bare regions | trained by `scripts/dataset/train_ppe_detector.py` on the two datasets below | AGPL-3.0 (derivative of Ultralytics weights) + CC BY 4.0 data | Yes - `models/`, Git LFS; reproducible | |
| Ultralytics Construction-PPE dataset (1,000 images, 12 classes incl. bare regions) | multi-PPE training + held-out test | platform.ultralytics.com (bundled with the ultralytics package) | CC BY 4.0 | No - downloaded by script | Attribution: Ultralytics team, Nov 2026, CC BY 4.0; its official val is our test split |
| HB1204/PPE_Detection dataset (3,086 images) | multi-PPE training (Phase 3) | huggingface.co/datasets/HB1204/PPE_Detection (Roboflow export) | CC BY 4.0 | No - downloaded by script | Roboflow-augmented; no "no boots" class |
| SH17 dataset (8,099 images) | NOT used (reserve) | zenodo.org/records/12659325 | CC BY-NC-SA 4.0 (non-commercial) | No - downloaded as reserve only | Non-commercial: must not be used for the employer-demo/training path; documented in docs/ml/phase3-log.md |
| Hard Hat Workers dataset (5,000 images; helmet/head/person) | fine-tuning + evaluation | huggingface.co/datasets/Voxel51/hard-hat-detection (originally Northeastern University – China via Roboflow / MakeML) | CC0-1.0 | No | Domain: construction close-ups, 416x416; not ship/industrial footage |
| Demo clips (10) | simulated camera streams | Pexels (IDs in `scripts/demo/demo_videos.yaml`) | Pexels License (free use/modify; no resale of unaltered copies; no implied endorsement by people shown) | No - fetched by script | Identifiable people appear; used only as simulated feeds, never for identification |
| MediaMTX 1.21.3, Toxiproxy 2.13.0 | RTSP simulator, fault injection | Docker images | MIT | No | Dev/demo only |
| PostgreSQL 17, Prometheus 3, Grafana 13 | services | Docker images | PostgreSQL / Apache-2.0 / AGPL-3.0 | No | |

### Phase 4 additions (2026-11-23, ADR-015; per-run detail in `docs/ml/phase4-log.md` §1)

| Asset | Used for | Source | Licence | Redistributed in this repo? | Notes |
|---|---|---|---|---|---|
| OmniFall (annotations + videos) | fall evaluation | OmniFall project | annotations CC BY-NC-SA 4.0; videos from their source datasets | No - `data/` git-ignored | **Eval only** (non-commercial annotations) |
| RLVS (1,000 Violence + 1,000 NonViolence clips) | fight GRU training (MVP) | Kaggle download | original authors' terms, **no explicit reuse licence** | No | MVP training only; retrain on licensed/client footage before commercial use |
| SCFD (150 fight / 150 noFight) | fight cross-dataset test | git clone, MIT code | code MIT; videos YouTube-sourced | No | **Test only** |
| COCO val2017 | phone-stage proxy eval | cocodataset.org | annotations CC BY 4.0; images under Flickr terms | No | **Eval only** |
| `fight_gru.pt` (ours) | fight classifier | `scripts/ml/train_fight.py` on RLVS | AGPL-3.0 code; training data **not** commercially licensed | Yes - `models/`, Git LFS | **MVP weights**: retrain before commercial deployment |

### Phase 5 additions (2026-11-24, ADR-016 revision)

| Asset | Used for | Source | Licence | Redistributed in this repo? | Notes |
|---|---|---|---|---|---|
| YuNet `face_detection_yunet_2023mar.onnx` | face detection at a door crossing (attendance identity) | github.com/opencv/opencv_zoo (ShiqiYu 2020) | MIT (zoo LICENSE, checked 2026-11-24) | No - downloaded into the models dir on first use (`app/attendance/faces.py`) | Runs through the already-installed OpenCV `FaceDetectorYN`; no new dependency |
| SFace `face_recognition_sface_2021dec.onnx` | face embedding for attendance identity | github.com/opencv/opencv_zoo | Apache-2.0 (zoo LICENSE for the model, checked 2026-11-24) | No - downloaded on first use | Frontally trained; angle-aware thresholds compensate, measured plan in ADR-016 |

## Conclusions
- Publishing this repository under AGPL-3.0 is compatible with every component above.
- A **closed-source or commercial deployment** would require an Ultralytics Enterprise licence or replacing the detector
  adapter with a permissively licensed model; it is **not represented as commercially deployable** as-is.
- Model outputs are assistive signals; no accuracy claim beyond the documented test-split and demo evaluations.
