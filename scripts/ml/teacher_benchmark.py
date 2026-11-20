"""Live benchmark of the teacher (Gemini Decisions API) on PUBLIC CC0 head crops only - never site footage.

Set: a balanced fixed-seed sample of the hard-hat test split + every test crop our classifier gets wrong.
Every teacher response is cached (cache.jsonl, keyed by image hash + model + question) so reruns never re-query;
every HTTP attempt is logged to requests.jsonl. Weight / thresholds are chosen on one half (hash split) and
reported on the other. Writes docs/ml/teacher_eval.json.

Run: cd backend && UV_PROJECT_ENVIRONMENT=... uv run python ../scripts/ml/teacher_benchmark.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.inference.teacher import INSTRUCTIONS, GeminiDecisionTeacher, decide, fuse  # noqa: E402

WORK = Path.home() / "ppe-work"
MAX_WRONG_TAG = 0.02  # tuning constraint: at most 2% of true alerts tagged LIKELY_FALSE_ALERT
MAX_FALSE_ESCALATE = 0.02  # ... and at most 2% of truly compliant samples escalated


def classifier_scores(test: Path, device: str) -> dict[str, tuple[int, float]]:
    """rel path -> (truth 1=no_helmet, our classifier's P(no_helmet)) over the WHOLE test split."""
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "head_cls_yolo26n.pt"))
    no_idx = next(k for k, v in model.names.items() if v == "no_helmet")
    out = {}
    for truth, cls in ((0, "helmet"), (1, "no_helmet")):
        files = sorted((test / cls).glob("*.jpg"))
        for i in range(0, len(files), 512):
            for f, r in zip(
                files[i : i + 512],
                model.predict([str(x) for x in files[i : i + 512]], imgsz=96, device=device, verbose=False),
                strict=True,
            ):
                out[f"{cls}/{f.name}"] = (truth, float(r.probs.data[no_idx]))
    return out


def cache_key(img: bytes, model: str) -> str:
    return hashlib.sha256(img + model.encode() + INSTRUCTIONS.encode()).hexdigest()


async def query_all(items: list[str], test: Path, cache_path: Path, req_log: Path, model: str, pace_s: float) -> None:
    cache = {json.loads(line)["key"] for line in cache_path.read_text().splitlines()} if cache_path.exists() else set()
    key = os.environ.get("GEMINI_API_KEY") or next(
        (
            ln.split("=", 1)[1].strip()
            for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("GEMINI_API_KEY=")
        ),
        "",
    )
    if not key:
        raise SystemExit("GEMINI_API_KEY missing")
    teacher = GeminiDecisionTeacher(key, model=model, timeout_s=30)
    todo = [r for r in items if cache_key((test / r).read_bytes(), model) not in cache]
    print(f"{len(items)} crops, {len(items) - len(todo)} cached, {len(todo)} to query", flush=True)
    last = 0.0
    for n, rel in enumerate(todo, 1):
        img = (test / rel).read_bytes()
        for attempt in range(8):
            await asyncio.sleep(max(0.0, last + pace_s - time.monotonic()))
            last = time.monotonic()
            status, rec = "ok", None
            try:
                v = await teacher.review(img)
                rec = {
                    "key": cache_key(img, model),
                    "file": rel,
                    "model": model,
                    "p_teacher": v.p_missing,
                    "latency_ms": round(v.latency_ms, 1),
                    "usage": v.raw.get("usage"),
                    "ts": time.time(),
                }
            except httpx.HTTPStatusError as e:
                status = str(e.response.status_code)
                if e.response.status_code == 400:  # e.g. safety block: record it, never retry
                    rec = {
                        "key": cache_key(img, model),
                        "file": rel,
                        "model": model,
                        "p_teacher": None,
                        "error": e.response.text[:300],
                        "ts": time.time(),
                    }
            except (httpx.TransportError, ValueError, KeyError) as e:
                status = type(e).__name__
            with req_log.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "file": rel, "status": status}) + "\n")
            if rec is not None:
                with cache_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                break
            wait = min(60.0, 2.0 * 2**attempt)  # 429 / 5xx / network: exponential backoff
            print(f"  {rel}: {status}, retry in {wait:.0f} s", flush=True)
            await asyncio.sleep(wait)
        else:
            raise SystemExit(f"giving up on {rel} after repeated failures - see {req_log}")
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- metrics


def auroc(y: list[int], p: list[float]) -> float:
    pos = [s for t, s in zip(y, p, strict=True) if t]
    neg = [s for t, s in zip(y, p, strict=True) if not t]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def ece(y: list[int], p: list[float], bins: int = 10) -> float:
    tot = 0.0
    for b in range(bins):
        idx = [i for i, s in enumerate(p) if b / bins <= s < (b + 1) / bins or (b == bins - 1 and s == 1.0)]
        if idx:
            tot += len(idx) / len(p) * abs(sum(p[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return tot


def scores(y: list[int], p: list[float]) -> dict:
    return {
        "accuracy": round(sum((s >= 0.5) == bool(t) for t, s in zip(y, p, strict=True)) / len(y), 4),
        "auroc": round(auroc(y, p), 4),
        "ece": round(ece(y, p), 4),
    }


def policy(rows: list[dict], w: float, t_esc: float, t_fa: float) -> dict:
    """Asymmetric policy. Model decision: p_model >= 0.5 => violation alert, else judged compliant."""
    c = {
        "missed_violations": 0,
        "escalated": 0,
        "false_alerts": 0,
        "false_alerts_tagged": 0,
        "true_alerts": 0,
        "true_alerts_wrongly_tagged": 0,
        "true_compliant": 0,
        "false_escalations": 0,
        "suppressed": 0,
    }
    for r in rows:
        kind = "violation" if r["p_model"] >= 0.5 else "compliant"
        v = decide(kind, fuse(r["p_teacher"], r["p_model"], w), t_esc, t_fa)
        c["suppressed"] += v not in ("CONFIRMED", "LIKELY_FALSE_ALERT", "ESCALATED", "COMPLIANT")
        if kind == "compliant" and r["truth"]:
            c["missed_violations"] += 1
            c["escalated"] += v == "ESCALATED"
        elif kind == "compliant":
            c["true_compliant"] += 1
            c["false_escalations"] += v == "ESCALATED"
        elif r["truth"]:
            c["true_alerts"] += 1
            c["true_alerts_wrongly_tagged"] += v == "LIKELY_FALSE_ALERT"
        else:
            c["false_alerts"] += 1
            c["false_alerts_tagged"] += v == "LIKELY_FALSE_ALERT"
    return c


def tune(rows: list[dict]) -> tuple[float, float, float]:
    """Per weight: lowest escalate / highest false-alert threshold within the 2% error budgets; keep the weight
    that catches the most of our errors (ties -> higher weight on our own model)."""
    grid = [i / 100 for i in range(1, 100)]
    best = None
    for w in [i / 20 for i in range(21)]:
        fused = [(r, fuse(r["p_teacher"], r["p_model"], w)) for r in rows]
        comp_neg = [f for r, f in fused if r["p_model"] < 0.5 and not r["truth"]]
        alert_pos = [f for r, f in fused if r["p_model"] >= 0.5 and r["truth"]]
        t_esc = min(t for t in grid + [1.0] if sum(f > t for f in comp_neg) <= MAX_FALSE_ESCALATE * len(comp_neg))
        t_fa = max(t for t in grid + [0.0] if sum(f < t for f in alert_pos) <= MAX_WRONG_TAG * len(alert_pos))
        c = policy(rows, w, t_esc, t_fa)
        caught = c["escalated"] + c["false_alerts_tagged"]
        if best is None or (caught, -w) > (best[0], -best[1]):
            best = (caught, w, t_esc, t_fa)
    assert best is not None
    return best[1], best[2], best[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(WORK / "hardhat" / "cls" / "test"))
    ap.add_argument("--cache", default=str(WORK / "teacher" / "cache.jsonl"))
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--model", default="gemini-decision-1")
    ap.add_argument("--pace-s", type=float, default=1.5, help="minimum seconds between requests (free tier)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--offline", action="store_true", help="analyse the cache only, no API calls")
    a = ap.parse_args()
    test, cache_path = Path(a.data), Path(a.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req_log = cache_path.with_name("requests.jsonl")

    cls = classifier_scores(test, a.device)
    wrong = sorted(r for r, (t, p) in cls.items() if (p >= 0.5) != bool(t))
    rng = random.Random(42)
    sample = set()
    for name in ("helmet", "no_helmet"):
        sample |= set(rng.sample(sorted(r for r in cls if r.startswith(name + "/")), a.per_class))
    items = sorted(sample | set(wrong))
    print(f"classifier errors on full test split: {len(wrong)} / {len(cls)}; benchmark set: {len(items)}", flush=True)
    if not a.offline:
        asyncio.run(query_all(items, test, cache_path, req_log, a.model, a.pace_s))

    recs = {}
    for line in cache_path.read_text().splitlines():
        r = json.loads(line)
        recs[r["file"]] = r
    rows, blocked = [], []
    for rel in items:
        r = recs.get(rel)
        if r is None or r.get("model") != a.model:
            continue
        if r.get("p_teacher") is None:
            blocked.append(rel)
            continue
        truth, p_model = cls[rel]
        h = int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16)
        rows.append(
            {
                "file": rel,
                "truth": truth,
                "p_model": p_model,
                "p_teacher": r["p_teacher"],
                "latency_ms": r["latency_ms"],
                "half": "tune" if h % 2 == 0 else "report",
            }
        )
    if not rows:
        raise SystemExit("no cached teacher responses - nothing to report")
    tune_rows = [r for r in rows if r["half"] == "tune"]
    rep_rows = [r for r in rows if r["half"] == "report"]
    w, t_esc, t_fa = tune(tune_rows)
    lat = sorted(r["latency_ms"] for r in rows)
    attempts = [json.loads(x) for x in req_log.read_text().splitlines()] if req_log.exists() else []
    status_counts: dict[str, int] = {}
    for x in attempts:
        status_counts[x["status"]] = status_counts.get(x["status"], 0) + 1

    def block(rs):
        y = [r["truth"] for r in rs]
        return {
            "n": len(rs),
            "no_helmet": sum(y),
            "teacher": scores(y, [r["p_teacher"] for r in rs]),
            "our_classifier": scores(y, [r["p_model"] for r in rs]),
            "fused_at_chosen_w": scores(y, [fuse(r["p_teacher"], r["p_model"], w) for r in rs]),
        }

    out = {
        "provider": "gemini",
        "model": a.model,
        "endpoint": "POST https://generativelanguage.googleapis.com/v1beta/"
        "decisions (x-goog-api-key header, noul question)",
        "question": INSTRUCTIONS,
        "data": "public CC0 hard-hat test split head crops only (oracle boxes); no site footage was sent",
        "set": {
            "balanced_per_class": a.per_class,
            "seed": 42,
            "classifier_errors_full_test_split": len(wrong),
            "classifier_test_split_n": len(cls),
            "benchmark_n": len(items),
            "with_teacher_answer": len(rows),
            "teacher_refused_400": len(blocked),
        },
        "note": "The set is enriched with ALL classifier errors, so error counts are not deployment rates.",
        "chosen_on_tune_half": {
            "weight": w,
            "escalate_threshold": t_esc,
            "false_alert_threshold": t_fa,
            "constraints": {
                "max_true_alerts_wrongly_tagged": MAX_WRONG_TAG,
                "max_compliant_falsely_escalated": MAX_FALSE_ESCALATE,
            },
        },
        "report_half": {**block(rep_rows), "policy": policy(rep_rows, w, t_esc, t_fa)},
        "tune_half": {**block(tune_rows), "policy": policy(tune_rows, w, t_esc, t_fa)},
        "all": block(rows),
        "latency_ms": {
            "p50": round(statistics.median(lat), 1),
            "p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "n": len(lat),
        },
        "requests": {
            "http_attempts_logged": len(attempts),
            "by_status": status_counts,
            "note": "plus 1 manual probe request before the benchmark (endpoint check)",
        },
    }
    dst = ROOT / "docs" / "ml" / "teacher_eval.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
