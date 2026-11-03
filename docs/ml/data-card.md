# Data card

## Hard Hat Workers (training / evaluation)
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
