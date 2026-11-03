# ADR-003: YOLO26 cascade (Ultralytics, AGPL-3.0): pose person model + fine-tuned helmet/head detector

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Need person and hard-hat detection with offline weights, good speed on one GPU, keypoints for a head ROI (FR-DET-005) and a
licence compatible with publishing the code.

## Decision
- Person stage: `yolo26s-pose` (COCO keypoints) - person boxes and 17 keypoints in one pass.
- PPE stage: `yolo26s` fine-tuned on **Hard Hat Workers** (CC0-1.0, 5,000 images, classes helmet/head/person; obtained from
  Voxel51/hard-hat-detection on Hugging Face). We use `helmet` and `head` (= a visible head without a hard hat).
- Optional head-crop classifier: `yolo26n-cls` trained on helmet/head crops of the same dataset; called only for ambiguous cases.
- Repository licensed **AGPL-3.0-or-later** (user decision 2026-11-02) so Ultralytics AGPL code and weights can be used.

## Alternatives considered
- YOLO-OBB (oriented boxes) for helmets, as proposed in the original whiteboard design: rejected for now - helmets are compact,
  roughly isotropic objects and axis-aligned boxes inside a head ROI suffice; adopt OBB only if measured failures on tilted
  helmets justify it (spec 6.4).
- Community pre-trained hard-hat checkpoints: unclear training-data licence and provenance.
- Apache/MIT detectors (RT-DETR via transformers, YOLOX): permissive, but slower to reach a good hard-hat model here.
- COCO person detector without pose: kept as a config option (`person.weights: yolo26s.pt`, `association.head_roi: geometry`).

## Trade-offs
AGPL obligations apply to anyone running a modified version as a network service; a closed commercial deployment would need an
Ultralytics Enterprise licence or a model swap (the Detector adapter keeps that change inside `app/inference`).

## Consequences
Weights are not committed; `models/README.md` documents how to reproduce them with `scripts/dataset/*`.

## Evidence / benchmark
Test split (500 images): helmet P 0.918 / R 0.938 / mAP50 0.964; head P 0.892 / R 0.922 / mAP50 0.937 (docs/ml/ppe_yolo26s_eval.json). Frozen-backbone ablation gave no gain (docs/ml/ppe_yolo26s_freeze10_eval.json). Classifier top-1 0.973 (docs/ml/head_classifier_eval.json). Cascade ablation and latency: docs/benchmarks/README.md. Observed limitation: caps/head wraps accepted as hard hats on demo footage (risk R16).

## Follow-up actions
Domain gap: the dataset is mostly construction close-ups at 416 px; collect site-specific footage before any operational claim.
