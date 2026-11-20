# Data card

## Ultralytics Construction-PPE + HB1204/PPE_Detection (Phase 3 multi-PPE detector)
- Ultralytics Construction-PPE: platform.ultralytics.com (archive bundled with the ultralytics package),
  **CC BY 4.0**; 1,000 images (800 train / 200 val), 1264-2048 px, 12 classes incl. all five items and their
  "no_*" bare-region classes; manifest.jsonl has per-image sha256. Its official val (200 images) is our **test**
  split, never trained on.
- HB1204/PPE_Detection: huggingface.co/datasets/HB1204/PPE_Detection (Roboflow export), **CC BY 4.0**;
  3,086 images (2,717 / 115 / 254), same item classes, no "no boots".
- Preparation: `scripts/dataset/prepare_multippe_dataset.py` → unified 10-class YOLO set (helmet, no_helmet, vest,
  no_vest, gloves, no_gloves, goggles, no_goggles, boots, no_boots); person/forklift dropped. Val = seeded 12.5 % of
  Construction-PPE train. Counts in the generated dataset_report.json.
- "No_*" convention: the body region is visible and bare; occlusion alone is never labelled absent (dataset README).
- Known gaps: few welding/indoor scenes, gloves/goggles annotations are sparse, HB1204 is Roboflow-augmented
  (augmentation note in its README); no site/video ids → leakage across near-duplicates cannot be ruled out.

## SH17 (downloaded, NOT used)
- zenodo.org/records/12659325, 8,099 Pexels images up to 8K, licence **CC BY-NC-SA 4.0 (non-commercial)** and a
  different absence convention (bare-hand/foot/face classes, not "no_*"). Reserve for future hand/foot regions.

## Hard Hat Workers (Phase 1 helmet stack - unchanged)
- Source: huggingface.co/datasets/Voxel51/hard-hat-detection (FiftyOne export of the Hard Hat Workers dataset,
  originally Northeastern University – China via Roboflow / MakeML). Licence **CC0-1.0**.
- 5,000 images, 416×416, classes helmet / head / person (VOC-style boxes).
- Preparation: `scripts/dataset/prepare_hardhat_dataset.py` → YOLO format + head-crop classification set.
- Split: images grouped by 8×8 average hash (4,831 groups) before a seeded 80/10/10 split → 3,976 / 524 / 500 images, to
  keep near-duplicates on one side (spec 12.2). The source has no video/site ids, so leakage cannot be ruled out fully.
- Label counts (train / val / test): helmet 15,012 / 2,083 / 1,871; head 4,654 / 538 / 593; person 603 / 78 / 70.
- Known gaps: few hard negatives (helmet in hand / on the ground), no maritime scenes, limited lighting variety.

## Demo footage (simulated cameras, not training data)
- 10 Pexels clips (IDs in `scripts/demo/demo_videos.yaml`), Pexels License, transcoded to 720p H.264 locally.
- Scene mix: 4 mostly without hard hats (factory/assembly/garment/masonry), 2 mixed sites, 4 compliant sites.
- Identifiable people appear; clips are used only as simulated camera feeds. No identity processing exists.
- Not redistributed; fetched by `scripts/demo/fetch_demo_videos.py`.
