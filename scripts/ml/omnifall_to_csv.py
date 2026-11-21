"""Export an OmniFall split to the eval_fall.py CSV (video,start,end,label).

Labels are CC BY-NC-SA 4.0 (evaluation only, never redistributed). Run in a throwaway env, never as a backend dep:
  uv run --with omnifall --with datasets --with requests python scripts/ml/omnifall_to_csv.py \
    --config of-itw --out data/fall/of_itw_test.csv
OF-ItW needs the OOPS videos prepared once (~45 GB streamed, ~2.6 GB kept): `omnifall prepare OOPS --yes`, or - while
oops.cs.columbia.edu is down (HTTP 404, Nov 2026) - omnifall._oops.extract_oops() on the byte-identical HF mirror
https://huggingface.co/datasets/pengxiang/oops/resolve/main/video_and_anns.tar.gz. OF-Syn videos auto-download.
API per the dataset card (huggingface.co/datasets/simplexsigil2/omnifall): if `add_video` changed, follow its
STRUCTURE.md - the only contract this script needs is an absolute `video` path per label row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="of-itw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import omnifall
    from datasets import load_dataset

    # download=True fetches redistributable videos (OF-Syn); strict=True fails instead of writing None paths
    ds = omnifall.add_video(
        load_dataset("simplexsigil2/omnifall", a.config, split=a.split), config=a.config, download=True, strict=True
    )
    names = ds.features["label"].names
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "start", "end", "label"])
        for r in ds:
            w.writerow([r["video"], r["start"], r["end"], names[r["label"]]])
    print(f"{len(ds)} rows -> {a.out}")


if __name__ == "__main__":
    main()
