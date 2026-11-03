# Model and data licence inventory (spec 13.2 release gate)

Checked 2026-11-02. Code licence, weight licence and training-data licence are assessed separately.

| Asset | Used for | Source | Licence | Redistributed in this repo? | Notes |
|---|---|---|---|---|---|
| Ultralytics library 8.4.198 | training + inference | pypi.org/project/ultralytics | AGPL-3.0 (Enterprise licence available) | No (dependency) | Repo is AGPL-3.0-or-later to stay compatible |
| `yolo26s-pose.pt`, `yolo26s.pt`, `yolo26n-cls.pt` | person+keypoints; PPE fine-tune base; classifier base | github.com/ultralytics/assets releases | AGPL-3.0 (Ultralytics) | No - downloaded into `models/` | COCO / ImageNet pre-training |
| `ppe_yolo26s.pt` (ours) | helmet/head detector | trained by `scripts/dataset/train_ppe_detector.py` | AGPL-3.0 (derivative of Ultralytics weights) + CC0 data | No - reproducible | |
| `head_cls_yolo26n.pt` (ours) | head-crop classifier | `scripts/dataset/train_head_classifier.py` | AGPL-3.0 + CC0 data | No - reproducible | |
| Hard Hat Workers dataset (5,000 images; helmet/head/person) | fine-tuning + evaluation | huggingface.co/datasets/Voxel51/hard-hat-detection (originally Northeastern University – China via Roboflow / MakeML) | CC0-1.0 | No | Domain: construction close-ups, 416x416; not ship/industrial footage |
| Demo clips (10) | simulated camera streams | Pexels (IDs in `scripts/demo/demo_videos.yaml`) | Pexels License (free use/modify; no resale of unaltered copies; no implied endorsement by people shown) | No - fetched by script | Identifiable people appear; used only as simulated feeds, never for identification |
| MediaMTX 1.21.3, Toxiproxy 2.13.0 | RTSP simulator, fault injection | Docker images | MIT | No | Dev/demo only |
| PostgreSQL 17, Prometheus 3, Grafana 13 | services | Docker images | PostgreSQL / Apache-2.0 / AGPL-3.0 | No | |

## Conclusions
- Publishing this repository under AGPL-3.0 is compatible with every component above.
- A **closed-source or commercial deployment** would require an Ultralytics Enterprise licence or replacing the detector
  adapter with a permissively licensed model; it is **not represented as commercially deployable** as-is.
- Model outputs are assistive signals; no accuracy claim beyond the documented test-split and demo evaluations.
