# Model card - PPE Intelligence detector stack v0.1

## Models
| Role | File | Base | Training | Version recorded at runtime |
|---|---|---|---|---|
| Person + 17 keypoints | `yolo26s-pose.pt` | Ultralytics YOLO26s-pose (COCO) | none (pre-trained) | `yolo26s-pose@a083adb42303` |
| Helmet / bare head | `ppe_yolo26s.pt` | YOLO26s (COCO) | full fine-tune, 60 epochs, 640 px, seed 42, Hard Hat Workers | `ppe_yolo26s@8f8ede05ecef` |
| Head-crop classifier | `head_cls_yolo26n.pt` | YOLO26n-cls | 30 epochs, 96 px, crops of helmet/head boxes | `head_cls_yolo26n@42b42b2de463` |

## Intended use
Assistive detection of people without hard hats on fixed industrial cameras, followed by temporal confirmation and
human review. **Not** for identification of individuals, disciplinary decisions, or safety-critical automation.

## Metrics (held-out TEST split, 500 images, near-duplicate-grouped split) - `docs/ml/*.json`
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.918 | 0.938 | 0.964 | 0.640 |
| head (no hard hat) | 0.892 | 0.922 | 0.937 | 0.622 |
| person (dataset class, unused) | 0.036 | 0.029 | 0.016 | 0.007 |

The dataset's `person` labels are sparse (≈ 1k vs 19k helmet/head), so persons come from the pose model instead.

Frozen-backbone ablation (freeze 10 layers): helmet mAP50 0.964 / mAP50-95 0.631, head 0.938 / 0.616, 24.7 min vs ~28 min
for the full fine-tune - no accuracy gain and slightly worse localisation, so the full fine-tune is used.

Head classifier (2,378 test crops): top-1 0.973; 53 of 576 bare heads (9.2 %) predicted "helmet" (the unsafe error
direction). At the runtime UNCERTAIN threshold 0.75: 97.6 % coverage, 98.2 % accuracy on kept crops. Crops come from
labelled boxes (oracle localisation) - runtime crops from the head ROI are noisier.

## Limitations (important)
- Training data are mostly construction-site close-ups at 416 px; the demo streams are 720p wide shots of factories and
  sites. **Image-level metrics above are not system-level event recall and say nothing about ship environments.**
- No night, rain, glare or maritime footage in training. **Observed on the demo streams: caps and head wraps are judged
  as hard hats** (metal-workshop camera produced 0 alerts although nobody wears a hard hat) - non-hard-hat headwear is
  absent from the training labels. Hard-negative data is the first improvement needed.
- The interview targets (≈80 % recall, ≤ 1 FP / 2 h) are **not** demonstrated: that requires labelled site footage and a
  documented observation period. The system reports FP per camera-hour from operator feedback to support that study.
