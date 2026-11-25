"""Retraining flywheel (ADR-018, docs/ml/flywheel.md): mine each check's false positives / negatives from the teacher
(VLM) reviews, sort them per check and label, notify the developer when a threshold is met, and after their approval
fine-tune the check's model on its old dataset + the approved frames. The new weights replace the old ones ONLY if
the gate passes: no regression on the developer-approved benchmark AND fewer errors on held-out mined frames.

  mine       teacher reviews -> <work>/<check>/inbox/<label>/ (FN = a violation the model passed, FP = false alarm);
             inbox >= threshold -> a review batch + contact sheet (review.jpg) + notification
  approve    the developer looked at the batch (move a file between label folders to relabel, delete it to drop)
  reject     the batch is never trained on
  benchmark  freeze the benchmark set + gate tolerances, signed by the developer (train refuses without it)
  train      champion fine-tuned on base train + approved frames -> candidate -> gate -> promote or keep champion
  run        mine, then train every check that has a newly approved batch (schedule this: cron / Task Scheduler)
  status     every check: trainer, inbox vs threshold, batches, benchmark

Run: cd backend && uv run python ../scripts/ml/flywheel.py <cmd> [--config ../configs/flywheel.yaml]
Self-check (no GPU, no DB): ... flywheel.py selftest
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from distill_finetune import link_tree, metrics, split  # noqa: E402 - same recipe + 70/30 hash split

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
HIGHER_IS_BETTER = {"accuracy", "auroc"}
APPROVED = {"approved", "promoted", "gate_failed"}  # developer-approved labels; all of them stay in the training pool


def now() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------- config + state


def load_config(path: str | None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "flywheel.yaml"
    if not p.exists():
        p = ROOT / "configs" / "flywheel.example.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["work_dir"] = Path(cfg["work_dir"]).expanduser()
    return cfg


def state_path(cfg: dict, check: str) -> Path:
    return cfg["work_dir"] / check / "state.json"


def load_state(cfg: dict, check: str) -> dict:
    p = state_path(cfg, check)
    return json.loads(p.read_text()) if p.exists() else {"seen": [], "batches": {}}


def save_state(cfg: dict, check: str, st: dict) -> None:
    p = state_path(cfg, check)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)  # atomic: a crash never leaves a half-written state


def notify(cfg: dict, text: str, **data) -> None:
    """Console + notifications.jsonl always; a Slack/Teams webhook if FLYWHEEL_WEBHOOK_URL is set (kept out of
    config files because the URL is a credential)."""
    print(f"[NOTIFY] {text}", flush=True)
    cfg["work_dir"].mkdir(parents=True, exist_ok=True)
    with open(cfg["work_dir"] / "notifications.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now(), "text": text, **data}) + "\n")
    url = os.environ.get("FLYWHEEL_WEBHOOK_URL")
    if url:
        req = urllib.request.Request(url, json.dumps({"text": text}).encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)  # noqa: S310 - developer-configured https webhook
        except OSError as e:  # the batch and its record are already on disk; a failed webhook loses nothing
            print(f"[NOTIFY] webhook failed: {type(e).__name__}", flush=True)


# ---------------------------------------------------------------- mining


def mistake(p_model: float, p_teacher: float | None, margin: float, feedback: str | None = None) -> str | None:
    """FN = the model passed a violation, FP = it alarmed on a compliant sample; None = agreement or unsure.
    Probabilities are P(violation). An operator decision on the event beats the teacher."""
    if feedback in ("TRUE_POSITIVE", "FALSE_POSITIVE"):
        truth = feedback == "TRUE_POSITIVE"
    elif p_teacher is not None and abs(p_teacher - 0.5) >= margin:
        truth = p_teacher >= 0.5
    else:
        return None
    said = p_model >= 0.5
    return None if truth == said else ("FN" if truth else "FP")


def scores(weights: Path, files: list[Path], c: dict, device: str) -> list[float]:
    """P(absent label) per crop from an ImageFolder classifier."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    absent = next(k for k, v in model.names.items() if v == c["labels"][1])
    out: list[float] = []
    for i in range(0, len(files), 512):
        rs = model.predict([str(f) for f in files[i:i + 512]], imgsz=c["imgsz"], device=device, verbose=False)
        out += [float(r.probs.data[absent]) for r in rs]
    return out


def from_db(cfg: dict) -> list[dict]:
    """Production source: every teacher review (all PPE checks) + operator feedback on the linked event."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.config.settings import get_settings
    from app.db.base import make_engine, make_session_factory, session_scope
    from app.models import Event, TeacherReview
    from sqlalchemy import select

    s = get_settings()
    root = s.resolve(s.ppe_evidence_dir)
    with session_scope(make_session_factory(make_engine(s.database_url))) as db:
        rows = db.execute(select(TeacherReview.id, TeacherReview.ppe_class, TeacherReview.crop_path,
                                 TeacherReview.p_model, TeacherReview.p_teacher, Event.feedback)
                          .outerjoin(Event, Event.id == TeacherReview.event_id)).all()
    out = []
    for rid, check, crop, pm, pt, fb in rows:
        kind = mistake(pm, pt, cfg["teacher_margin"], fb)
        src = (root / crop).resolve()
        if kind and src.is_relative_to(root) and src.is_file():  # crops expire with evidence retention
            out.append({"id": f"review{rid}", "check": check, "src": src, "p_model": pm, "p_teacher": pt,
                        "kind": kind, "source": "db"})
    return out


def from_cache(cfg: dict) -> list[dict]:
    """Demo source: cached real Gemini Decisions answers on public test crops; p_model from the current champion.
    The crops' folder is ground truth, recorded as `truth` so the demo can show how often the teacher was right."""
    c, dc = cfg["checks"]["hard_hat"], cfg["demo_cache"]
    rows = [json.loads(x) for x in Path(dc["path"]).expanduser().read_text().splitlines()]
    files = [Path(dc["data"]).expanduser() / r["file"] for r in rows]
    pm = scores(MODELS / c["weights"], files, c, cfg["device"])
    out = []
    for r, f, p in zip(rows, files, pm, strict=True):
        kind = mistake(p, r.get("p_teacher"), cfg["teacher_margin"])
        if kind:
            out.append({"id": f.stem, "check": "hard_hat", "src": f, "p_model": round(p, 4),
                        "p_teacher": round(r["p_teacher"], 4), "kind": kind, "source": "cache",
                        "truth": f.parent.name})
    return out


def place(cfg: dict, samples: list[dict]) -> dict[str, int]:
    """Sort new mistakes into <check>/inbox/<label>/ (deduped); open a review batch per check at its threshold."""
    added: Counter = Counter()
    states: dict[str, dict] = {}
    for s in samples:
        c = cfg["checks"].get(s["check"])
        if c is None:
            continue
        st = states.setdefault(s["check"], load_state(cfg, s["check"]))
        if s["id"] in st["seen"]:
            continue
        label = c["labels"][1] if s["kind"] == "FN" else c["labels"][0]  # the label the teacher says is right
        inbox = cfg["work_dir"] / s["check"] / "inbox"
        (inbox / label).mkdir(parents=True, exist_ok=True)
        name = f"{s['kind']}_{s['id']}{s['src'].suffix}"
        shutil.copy2(s["src"], inbox / label / name)
        man_p = inbox / "manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else {}
        man[f"{label}/{name}"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()}
        man_p.write_text(json.dumps(man, indent=1))
        st["seen"].append(s["id"])
        added[s["check"]] += 1
    for check, st in states.items():
        maybe_open_batch(cfg, check, st)
        save_state(cfg, check, st)
    return dict(added)


def maybe_open_batch(cfg: dict, check: str, st: dict) -> str | None:
    inbox = cfg["work_dir"] / check / "inbox"
    man_p = inbox / "manifest.json"
    if not man_p.exists():
        return None
    man = json.loads(man_p.read_text())
    if len(man) < cfg["checks"][check]["threshold"]:
        return None
    bid = f"{check}-{now()}"
    dst = cfg["work_dir"] / check / "batches" / bid
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(inbox), str(dst))
    contact_sheet(dst, man)
    kinds = Counter(v["kind"] for v in man.values())
    st["batches"][bid] = {"status": "pending_review", "created": now(), "n": len(man), "kinds": dict(kinds)}
    notify(cfg, f"{check}: {len(man)} mined frames ({kinds['FN']} missed violations, {kinds['FP']} false alarms) "
                f"need a look - {dst / 'review.jpg'}; then flywheel.py approve --check {check} --batch {bid}",
           check=check, batch=bid, path=str(dst))
    return bid


def contact_sheet(batch: Path, man: dict, cols: int = 10, px: int = 112) -> None:
    """One image the developer scans in seconds: red = FN (model missed it), blue = FP (false alarm)."""
    from PIL import Image, ImageDraw, ImageOps

    items = sorted(man.items(), key=lambda kv: (kv[1]["kind"], kv[0]))[:200]
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * px, rows * (px + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (rel, v) in enumerate(items):
        x, y = (i % cols) * px, (i // cols) * (px + 16)
        im = ImageOps.contain(Image.open(batch / rel).convert("RGB"), (px - 4, px - 4))  # up- or down-scale
        sheet.paste(im, (x + 2, y + 2))
        col = (200, 0, 0) if v["kind"] == "FN" else (0, 70, 200)
        d.rectangle([x, y, x + px - 1, y + px - 1], outline=col, width=2)
        d.text((x + 3, y + px), f"{v['kind']} m{v['p_model']:.2f} t{v['p_teacher']:.2f}", fill=col)
    sheet.save(batch / "review.jpg", quality=90)


# ---------------------------------------------------------------- developer decisions


def decide_batch(cfg: dict, check: str, bid: str, status: str, by: str, note: str = "") -> dict:
    st = load_state(cfg, check)
    b = st["batches"].get(bid)
    if b is None or b["status"] != "pending_review":
        sys.exit(f"batch {bid} is not pending review ({b and b['status']})")
    batch = cfg["work_dir"] / check / "batches" / bid
    labels = cfg["checks"][check]["labels"]
    counts = {lab: len(list((batch / lab).glob("*.*"))) for lab in labels}  # the folders are the truth after review
    b.update(status=status, decided_by=by, decided_at=now(), note=note, approved_counts=counts)
    save_state(cfg, check, st)
    print(json.dumps({bid: b}, indent=1))
    return b


def bench_files(c: dict) -> list[Path]:
    test = Path(c["base_data"]).expanduser() / "test"
    return sorted(f for lab in c["labels"] for f in (test / lab).glob("*.jpg"))


def files_sha(files: list[Path]) -> str:
    return hashlib.sha256("\n".join(f"{f.parent.name}/{f.name}" for f in files).encode()).hexdigest()[:16]


def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def evaluate(weights: Path, rows: list[tuple[Path, str]], c: dict, device: str) -> dict:
    p = scores(weights, [f for f, _ in rows], c, device)
    m = metrics([int(lab == c["labels"][1]) for _, lab in rows], p, 0.5)
    m["errors_at_0.5"] = m["missed_violations_at_0.5"] + m["false_alarms_at_0.5"]
    return m


def approve_benchmark(cfg: dict, check: str, by: str) -> dict:
    c = cfg["checks"][check]
    files = bench_files(c)
    w = MODELS / c["weights"]
    bench = {"approved_by": by, "approved_at": now(), "n": len(files), "files_sha": files_sha(files),
             "tolerances": cfg["gate"]["tolerances"], "mined_min_gain": cfg["gate"]["mined_min_gain"],
             "champion_at_approval": {"sha256_12": sha12(w),
                                      **evaluate(w, [(f, f.parent.name) for f in files], c, cfg["device"])}}
    p = cfg["work_dir"] / check / "benchmark.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bench, indent=1))
    print(json.dumps(bench, indent=1))
    return bench


# ---------------------------------------------------------------- train + gate + promote


def gate(champ: dict, cand: dict, tolerances: dict, mined_champ: int, mined_cand: int, min_gain: int):
    """Pass only if the candidate is no worse than the champion on every approved benchmark metric (within its
    tolerance) AND makes at least `min_gain` fewer errors on the held-out mined frames."""
    lines = []
    for k, tol in tolerances.items():
        a, b = champ[k], cand[k]
        ok = b >= a - tol if k in HIGHER_IS_BETTER else b <= a + tol
        lines.append((ok, f"benchmark {k}: champion {a} -> candidate {b} (tolerance {tol})"))
    lines.append((mined_champ - mined_cand >= min_gain,
                  f"held-out mined frames errors: champion {mined_champ} -> candidate {mined_cand} "
                  f"(must fix >= {min_gain})"))
    return all(ok for ok, _ in lines), [("PASS " if ok else "FAIL ") + t for ok, t in lines]


def train(cfg: dict, check: str, epochs: int | None = None) -> dict | None:
    c = cfg["checks"][check]
    if c["trainer"] != "classify":
        # shortcut: only the classify adapter is wired; detect/sequence batches stay approved and queued until their
        # adapter lands (docs/ml/flywheel.md "Trainer adapters"); rule checks have no weights to train.
        print(f"{check}: trainer '{c['trainer']}' not wired - approved batches stay queued")
        return None
    st = load_state(cfg, check)
    bench_p = cfg["work_dir"] / check / "benchmark.json"
    if not bench_p.exists():
        sys.exit(f"{check}: no approved benchmark - run: flywheel.py benchmark --check {check} --approve")
    bench = json.loads(bench_p.read_text())
    bfiles = bench_files(c)
    if files_sha(bfiles) != bench["files_sha"]:
        sys.exit(f"{check}: the benchmark set changed since it was approved - re-approve it")
    pool_ids = [b for b, v in st["batches"].items() if v["status"] in APPROVED]
    if not any(st["batches"][b]["status"] == "approved" for b in pool_ids):
        print(f"{check}: no newly approved batch - nothing to train")
        return None

    run = now()
    work = cfg["work_dir"] / check
    pool = [(f, lab) for b in pool_ids for lab in c["labels"] for f in (work / "batches" / b / lab).glob("*.jpg")]
    ft = [r for r in pool if split(r[0].name) == "finetune"]
    ho = [r for r in pool if split(r[0].name) == "holdout"]  # the mined-mistake test set: never trained on
    data = work / "runs" / run / "data"
    base = Path(c["base_data"]).expanduser()
    link_tree(base / "train", data / "train")
    link_tree(base / "val", data / "val")
    for f, lab in ft:
        os.link(f, data / "train" / lab / f"mined_{f.name}")

    from ultralytics import YOLO

    champion = MODELS / c["weights"]
    YOLO(str(champion)).train(data=str(data), epochs=epochs or c["epochs"], imgsz=c["imgsz"], batch=256,
                              device=cfg["device"], seed=42, deterministic=True, project=str(work / "runs" / run),
                              name="train", exist_ok=True, patience=10, workers=8, verbose=False)
    cand = work / "runs" / run / "train" / "weights" / "best.pt"

    pooled = {f.name for f, _ in pool} | {f.name.split("_", 1)[1] for f, _ in pool}
    brows = [(f, f.parent.name) for f in bfiles if f.name not in pooled]  # a mined crop never scores the benchmark
    ev = {k: {"benchmark": evaluate(w, brows, c, cfg["device"]), "mined_holdout": evaluate(w, ho, c, cfg["device"])}
          for k, w in (("champion", champion), ("candidate", cand))} if ho else None
    passed, lines = (gate(ev["champion"]["benchmark"], ev["candidate"]["benchmark"], bench["tolerances"],
                          ev["champion"]["mined_holdout"]["errors_at_0.5"],
                          ev["candidate"]["mined_holdout"]["errors_at_0.5"], bench["mined_min_gain"])
                     if ev else (False, ["FAIL no held-out mined frames - approve more frames first"]))
    report = {"check": check, "run": run, "batches": pool_ids, "train_mined": len(ft), "holdout_mined": len(ho),
              "benchmark_n": len(brows), "benchmark_excluded_overlap": len(bfiles) - len(brows),
              "champion_sha256_12": sha12(champion), "candidate_sha256_12": sha12(cand), "passed": passed,
              "gate": lines, "eval": ev}
    if passed:
        archive = MODELS / "archive" / f"{champion.stem}.{report['champion_sha256_12']}{champion.suffix}"
        archive.parent.mkdir(exist_ok=True)
        shutil.copy2(champion, archive)  # the old weights stay restorable before anything is overwritten
        tmp = champion.with_name(f".{champion.name}.tmp")
        shutil.copy2(cand, tmp)
        os.replace(tmp, champion)  # atomic on one filesystem: readers see old or new weights, never half a file
        report["archived_champion"] = archive.relative_to(ROOT).as_posix()
        with open(MODELS / "registry.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": run, "check": check, "weights": c["weights"],
                                "promoted": report["candidate_sha256_12"], "replaced": report["champion_sha256_12"],
                                "archive": report["archived_champion"], "batches": pool_ids,
                                "approved_benchmark_by": bench["approved_by"]}) + "\n")
    for b in pool_ids:
        if st["batches"][b]["status"] == "approved":
            st["batches"][b]["status"] = "promoted" if passed else "gate_failed"
            st["batches"][b]["run"] = run
    save_state(cfg, check, st)
    (work / "reports").mkdir(exist_ok=True)
    (work / "reports" / f"{run}.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))
    notify(cfg, f"{check}: candidate {'PROMOTED - restart the app to load it' if passed else 'REJECTED by the gate'}"
                f" (report {work / 'reports' / f'{run}.json'})", check=check, run=run, passed=passed)
    return report


def status(cfg: dict) -> list[dict]:
    rows = []
    for check, c in cfg["checks"].items():
        st = load_state(cfg, check)
        man_p = cfg["work_dir"] / check / "inbox" / "manifest.json"
        rows.append({"check": check, "trainer": c["trainer"],
                     "inbox": f"{len(json.loads(man_p.read_text())) if man_p.exists() else 0}/{c['threshold']}",
                     "batches": dict(Counter(b["status"] for b in st["batches"].values())),
                     "benchmark": (cfg["work_dir"] / check / "benchmark.json").exists(),
                     "weights": c.get("weights") or "- (rule)"})
    for r in rows:
        bench = "yes" if r["benchmark"] else "no"
        print(f"{r['check']:13} {r['trainer']:9} inbox {r['inbox']:>8}  benchmark {bench:3}  {r['weights']:22} "
              f"{r['batches'] or ''}")
    return rows


# ---------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mine", "run"):
        p = sub.add_parser(name)
        p.add_argument("--source", choices=["db", "cache"], default="db")
        if name == "run":
            p.add_argument("--epochs", type=int)
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("--check", required=True)
        p.add_argument("--batch", required=True)
        p.add_argument("--by", default=getpass.getuser())
        p.add_argument("--note", default="")
    p = sub.add_parser("benchmark")
    p.add_argument("--check", required=True)
    p.add_argument("--approve", action="store_true", required=True)
    p.add_argument("--by", default=getpass.getuser())
    p = sub.add_parser("train")
    p.add_argument("--check", required=True)
    p.add_argument("--epochs", type=int)
    sub.add_parser("status")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return 0
    cfg = load_config(a.config)
    if a.cmd in ("mine", "run"):
        print(json.dumps({"mined": place(cfg, from_db(cfg) if a.source == "db" else from_cache(cfg))}))
        if a.cmd == "run":
            for check in cfg["checks"]:
                if any(b["status"] == "approved" for b in load_state(cfg, check)["batches"].values()):
                    train(cfg, check, a.epochs)
    elif a.cmd in ("approve", "reject"):
        decide_batch(cfg, a.check, a.batch, "approved" if a.cmd == "approve" else "rejected", a.by, a.note)
    elif a.cmd == "benchmark":
        approve_benchmark(cfg, a.check, a.by)
    elif a.cmd == "train":
        train(cfg, a.check, a.epochs)
    else:
        status(cfg)
    return 0


def selftest() -> None:
    # mining: teacher / operator vs model, P(violation) everywhere
    assert mistake(0.1, 0.95, 0.3) == "FN" and mistake(0.9, 0.05, 0.3) == "FP"
    assert mistake(0.9, 0.95, 0.3) is None and mistake(0.1, 0.05, 0.3) is None  # agreement
    assert mistake(0.1, 0.7, 0.3) is None  # teacher unsure -> never mined
    assert mistake(0.9, 0.95, 0.3, "FALSE_POSITIVE") == "FP"  # operator beats teacher
    assert mistake(0.1, 0.05, 0.3, "TRUE_POSITIVE") == "FN"
    # gate: no regression on the benchmark AND a real gain on held-out mistakes
    tol = {"accuracy": 0.0, "missed_violations_at_0.5": 0}
    ok, _ = gate({"accuracy": 0.95, "missed_violations_at_0.5": 10}, {"accuracy": 0.95, "missed_violations_at_0.5": 9},
                 tol, 12, 5, 1)
    assert ok
    ok, lines = gate({"accuracy": 0.95, "missed_violations_at_0.5": 10},
                     {"accuracy": 0.949, "missed_violations_at_0.5": 9}, tol, 12, 5, 1)
    assert not ok and lines[0].startswith("FAIL")  # a tiny benchmark regression blocks promotion
    assert not gate({"accuracy": 0.9}, {"accuracy": 0.99}, {"accuracy": 0.0}, 5, 5, 1)[0]  # no gain on mined frames
    # sorting + threshold + batch + approval on a temp work dir
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        cfg = {"work_dir": Path(d) / "fw", "checks": {"hard_hat": {"labels": ["helmet", "no_helmet"], "threshold": 3,
                                                                   "trainer": "classify"}}}
        srcs = []
        for i in range(3):
            srcs.append(Path(d) / f"c{i}.jpg")
            Image.new("RGB", (32, 32), (i * 80, 0, 0)).save(srcs[-1])
        s = [{"id": f"c{i}", "check": "hard_hat", "src": p, "p_model": 0.1, "p_teacher": 0.9, "kind": "FN"}
             for i, p in enumerate(srcs[:2])]
        assert place(cfg, s) == {"hard_hat": 2} and place(cfg, s) == {}  # deduped
        assert not load_state(cfg, "hard_hat")["batches"]  # 2 < threshold 3
        place(cfg, [{"id": "c2", "check": "hard_hat", "src": srcs[2], "p_model": 0.8, "p_teacher": 0.1, "kind": "FP"}])
        (bid, b), = load_state(cfg, "hard_hat")["batches"].items()
        batch = cfg["work_dir"] / "hard_hat" / "batches" / bid
        assert b["status"] == "pending_review" and b["kinds"] == {"FN": 2, "FP": 1}
        assert (batch / "review.jpg").exists() and len(list((batch / "no_helmet").glob("*.jpg"))) == 2
        assert (cfg["work_dir"] / "notifications.jsonl").exists()
        (batch / "no_helmet" / "FN_c0.jpg").unlink()  # the developer drops one frame
        assert decide_batch(cfg, "hard_hat", bid, "approved", "dev")["approved_counts"] == {"helmet": 1, "no_helmet": 1}
    print("selftest ok")


if __name__ == "__main__":
    sys.exit(main())
