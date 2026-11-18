"""Distillation flywheel: export confident teacher labels as an ImageFolder dataset {helmet,no_helmet} to retrain
our own head classifier (scripts/dataset/train_head_classifier.py), so external teacher calls can be switched off.

Label rule: an operator decision on the linked event wins (TRUE_POSITIVE => no_helmet, FALSE_POSITIVE => helmet);
otherwise the teacher label when |p_teacher - 0.5| >= margin; anything else is skipped.

  --source db     teacher_reviews rows + crops from the evidence root (needs the app .env / database)
  --source cache  the benchmark cache (scripts/ml/teacher_benchmark.py) + the public test-split images

Also writes <out>/manifest.json: exported path -> {crop (source rel path), label, p_teacher, source}.

Run: cd backend && UV_PROJECT_ENVIRONMENT=... uv run python ../scripts/ml/export_teacher_labels.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
WORK = Path.home() / "ppe-work"


def label(p_teacher: float | None, margin: float, feedback: str | None = None) -> str | None:
    if feedback == "TRUE_POSITIVE":
        return "no_helmet"
    if feedback == "FALSE_POSITIVE":
        return "helmet"
    if p_teacher is None or abs(p_teacher - 0.5) < margin:
        return None
    return "no_helmet" if p_teacher >= 0.5 else "helmet"


def from_db(margin: float) -> list[tuple[Path, str, str, float | None, str]]:
    from app.config.settings import get_settings
    from app.db.base import make_engine, make_session_factory, session_scope
    from app.models import Event, TeacherReview
    from sqlalchemy import select

    s = get_settings()
    evidence_root = s.resolve(s.ppe_evidence_dir)
    sf = make_session_factory(make_engine(s.database_url))
    with session_scope(sf) as db:
        rows = db.execute(
            select(TeacherReview.id, TeacherReview.crop_path, TeacherReview.p_teacher, Event.feedback).outerjoin(
                Event, Event.id == TeacherReview.event_id
            )
        ).all()
    out = []
    for rid, crop, p, fb in rows:
        lab = label(p, margin, fb)
        src = (evidence_root / crop).resolve()
        if lab and src.is_relative_to(evidence_root) and src.is_file():  # crops expire with evidence retention
            out.append((src, lab, f"review{rid}.jpg", p, crop))
    return out


def from_cache(margin: float, cache: Path, data: Path) -> list[tuple[Path, str, str, float | None, str]]:
    out = []
    for line in cache.read_text().splitlines():
        r = json.loads(line)
        lab = label(r.get("p_teacher"), margin)
        if lab:
            out.append((data / r["file"], lab, Path(r["file"]).name, r["p_teacher"], r["file"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db", "cache"], default="db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.4, help="keep labels with |p - 0.5| >= margin")
    ap.add_argument("--cache", default=str(WORK / "teacher" / "cache.jsonl"))
    ap.add_argument("--data", default=str(WORK / "hardhat" / "cls" / "test"))
    a = ap.parse_args()
    items = from_db(a.margin) if a.source == "db" else from_cache(a.margin, Path(a.cache), Path(a.data))
    out = Path(a.out)
    counts = {"helmet": 0, "no_helmet": 0}
    manifest = {}
    for src, lab, name, p, crop in items:
        (out / lab).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / lab / name)
        counts[lab] += 1
        manifest[f"{lab}/{name}"] = {"crop": crop, "label": lab, "p_teacher": p, "source": a.source}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"source": a.source, "margin": a.margin, "exported": counts, "out": str(out)}))
    return 0


if __name__ == "__main__":
    assert label(0.95, 0.4) == "no_helmet" and label(0.05, 0.4) == "helmet" and label(0.7, 0.4) is None
    assert label(0.05, 0.4, "TRUE_POSITIVE") == "no_helmet" and label(None, 0.4, "FALSE_POSITIVE") == "helmet"
    sys.exit(main())
