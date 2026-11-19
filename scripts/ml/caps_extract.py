"""caps fix, step 1-2: extract head crops from the cam06 metal-workshop clip exactly like production, and
teacher-label them with the same Gemini Decisions question the app uses (stock Pexels footage - allowed).

Extraction: pose model -> head_roi() (keypoints mode with geometry fallback, the production ROI) with the same
not-assessable gates, crop like Associator._crop, save JPEG q85. Identity = greedy IoU matching of person boxes
across sampled frames (track-ish). Near-identical crops of one identity are deduped by colour-histogram
correlation so the teacher bills once per distinct worker/pose.

Labeling reuses teacher_benchmark.query_all() unchanged (cache, 1.5 s pace, 429 backoff, 400 blocks recorded).
Cache: C:/Users/mianx/ppe-work/caps/cache.jsonl. models/ is never modified.

Run: cd backend && UV_PROJECT_ENVIRONMENT=... uv run python ../scripts/ml/caps_extract.py --label
Self-check: ... caps_extract.py --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config.settings import AssociationConfig, RuleThresholds  # noqa: E402
from app.ppe.association import Associator  # noqa: E402
from app.ppe.head_roi import head_roi  # noqa: E402
from app.ppe.types import TrackedPerson  # noqa: E402

WORK = Path.home() / "ppe-work" / "caps"
CLIP = Path.home() / "ppe-work" / "streams" / "cam06.mp4"
IOU_MATCH = 0.3  # person-box IoU above which a detection continues an existing identity
DEDUPE_CORR = 0.985  # histogram correlation above which a crop is a near-duplicate of a kept one
JPEG_Q = 85  # production EvidenceConfig.jpeg_quality


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign_identities(boxes: list[tuple[float, float, float, float]],
                      tracks: list[tuple[float, tuple[float, float, float, float]]]) -> list[int]:
    """Greedy IoU matching against the previous frame's tracks; returns one track id per box (new ids appended).
    tracks is mutated: each matched track's box is replaced by the detection's."""
    pairs = sorted(((iou(b, tb), ti, bi) for bi, b in enumerate(boxes)
                    for ti, (_, tb) in enumerate(tracks) if iou(b, tb) >= IOU_MATCH), reverse=True)
    out = [-1] * len(boxes)
    used_t: set[int] = set()
    for _s, ti, bi in pairs:
        if out[bi] >= 0 or ti in used_t:
            continue
        out[bi] = ti
        used_t.add(ti)
    for bi, b in enumerate(boxes):
        if out[bi] < 0:
            tracks.append((len(tracks), b))
            out[bi] = tracks[-1][0]
    for bi, ti in enumerate(out):  # refresh matched tracks' boxes with the latest detection
        tracks[ti] = (ti, boxes[bi])
    return out


def hist(image_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(image_bgr, (96, 96))
    h = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h


def extract(clip: Path, out: Path, sample_fps: float, device: str) -> list[dict]:
    from ultralytics import YOLO

    cfg, thr = AssociationConfig(), RuleThresholds()
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    pose = YOLO(str(ROOT / "models" / "yolo26s-pose.pt"))
    rows: list[dict] = []
    tracks: list[tuple[float, tuple[float, float, float, float]]] = []
    kept_hists: dict[int, list[np.ndarray]] = {}
    n_frames = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n_frames % step:
            n_frames += 1
            continue
        r = pose.predict(img, imgsz=640, conf=0.25, device=device, classes=[0], verbose=False)[0]
        persons, boxes = [], []
        kdata = None if r.keypoints is None else r.keypoints.data.cpu().numpy()
        for pi in range(len(r.boxes)):
            conf = float(r.boxes.conf[pi])
            box = tuple(float(v) for v in r.boxes.xyxy[pi])
            if conf < thr.person_conf or box[3] - box[1] < thr.min_person_height_px or box[1] <= 2.0:
                continue  # production _not_assessable gates (low conf / too small / head truncated)
            persons.append(TrackedPerson(track_id=pi, box=box, conf=conf,
                                         keypoints=None if kdata is None else kdata[pi]))
            boxes.append(box)
        ids = assign_identities(boxes, tracks)
        for p, ti in zip(persons, ids, strict=True):
            roi, src = head_roi(p, cfg, img.shape[1], img.shape[0])
            if roi[2] - roi[0] < thr.min_head_px or roi[3] - roi[1] < thr.min_head_px:
                continue
            crop = Associator._crop(img, roi)
            h = hist(crop)
            if any(float(cv2.compareHist(h, kh, cv2.HISTCMP_CORREL)) >= DEDUPE_CORR for kh in kept_hists.get(ti, [])):
                continue
            kept_hists.setdefault(ti, []).append(h)
            name = f"tr{ti:03d}_f{n_frames:05d}.jpg"
            cv2.imwrite(str(out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            rows.append({"file": name, "track": ti, "frame": n_frames, "t_sec": round(n_frames / fps, 3),
                         "roi_source": src, "person_conf": round(p.conf, 3)})
        n_frames += 1
    cap.release()
    return rows


def label_all(items: list[str], raw: Path, cache_path: Path, model: str) -> None:
    from teacher_benchmark import query_all

    asyncio.run(query_all(items, raw, cache_path, cache_path.with_name("requests.jsonl"), model, 1.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=str(CLIP))
    ap.add_argument("--out", default=str(WORK))
    ap.add_argument("--sample-fps", type=float, default=3.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default="gemini-decision-1")
    ap.add_argument("--label", action="store_true", help="also teacher-label the extracted crops")
    ap.add_argument("--force", action="store_true", help="re-extract even if raw crops exist")
    a = ap.parse_args()
    out, raw = Path(a.out), Path(a.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    rows = json.loads(manifest.read_text())["crops"] if manifest.exists() and not a.force else None
    if rows is None:
        rows = extract(Path(a.clip), raw, a.sample_fps, a.device)
        manifest.write_text(json.dumps({"clip": a.clip, "sample_fps": a.sample_fps, "crops": rows}, indent=1))
    print(json.dumps({"crops": len(rows), "identities": len({r["track"] for r in rows}),
                      "roi_sources": {s: sum(r["roi_source"] == s for r in rows) for s in ("keypoints", "geometry")}}))

    cache_path = out / "cache.jsonl"
    if a.label:
        label_all(sorted(r["file"] for r in rows), raw, cache_path, a.model)
    p_by_key = {}
    if cache_path.exists():
        from teacher_benchmark import cache_key

        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            p_by_key[r["key"]] = r.get("p_teacher")
    for r in rows:
        key = cache_key((raw / r["file"]).read_bytes(), a.model) if cache_path.exists() else None
        r["p_teacher"] = p_by_key.get(key)
    manifest.write_text(json.dumps({"clip": a.clip, "sample_fps": a.sample_fps, "crops": rows}, indent=1))
    got = [r["p_teacher"] for r in rows if r["p_teacher"] is not None]
    print(json.dumps({"labeled": len(got), "blocked": len(rows) - len(got),
                      "mean_p": round(sum(got) / len(got), 4) if got else None}))
    return 0


def selftest() -> None:
    mk = lambda x1, y1, x2, y2: (float(x1), float(y1), float(x2), float(y2))  # noqa: E731
    assert iou(mk(0, 0, 10, 10), mk(0, 0, 10, 10)) == 1.0 and iou(mk(0, 0, 10, 10), mk(20, 20, 30, 30)) == 0.0
    assert abs(iou(mk(0, 0, 10, 10), mk(5, 0, 15, 10)) - 50 / 150) < 1e-9
    tracks: list = []
    ids = assign_identities([mk(0, 0, 10, 20)], tracks)
    assert ids == [0] and len(tracks) == 1
    ids = assign_identities([mk(1, 0, 11, 20), mk(50, 0, 60, 20)], tracks)  # one continues, one is new
    assert ids == [0, 1] and len(tracks) == 2
    assert assign_identities([mk(50, 0, 60, 20), mk(1, 0, 11, 20)], tracks) == [1, 0]  # order-independent
    assert assign_identities([mk(200, 0, 210, 20)], tracks) == [2]  # no overlap => new identity
    t: list = []  # deterministic across calls: same boxes keep the same identities
    assert [assign_identities([mk(0, 0, 10, 20)], t)[0] for _ in range(3)] == [0, 0, 0]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    sys.exit(main())
