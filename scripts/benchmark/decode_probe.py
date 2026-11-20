"""Phase 3 decode capacity probe: how many 4K H.264 streams can one host decode, CPU vs NVDEC (PyAV hwaccel)?

Decodes N copies of a clip in parallel threads (one PyAV container each, like CameraSource) for --seconds and
converts every k-th frame to BGR (the pipeline converts at 2 x target fps, i.e. every 2.5th frame at 25 fps / 5 fps).
Reports decoded fps per stream, aggregate, and host CPU %. Writes docs/benchmarks/results/decode_<tag>.json.

  python scripts/benchmark/decode_probe.py --clip <4k.mp4> --streams 1 4 10 --hw none cuda
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import av
import psutil
from av.codec.hwaccel import HWAccel

ROOT = Path(__file__).resolve().parents[2]


def worker(clip: str, hw: str, stop: threading.Event, convert_every: int, out: dict, i: int) -> None:
    kw = {"hwaccel": HWAccel(device_type=hw, allow_software_fallback=False)} if hw != "none" else {}
    n = conv = 0
    with av.open(clip, **kw) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        while not stop.is_set():
            for f in c.decode(s):
                n += 1
                if n % convert_every == 0:
                    f.to_ndarray(format="bgr24")
                    conv += 1
                if stop.is_set():
                    break
            c.seek(0)
    out[i] = (n, conv)


def run(clip: str, streams: int, hw: str, seconds: float, convert_every: int) -> dict:
    stop, out = threading.Event(), {}
    ts = [threading.Thread(target=worker, args=(clip, hw, stop, convert_every, out, i)) for i in range(streams)]
    psutil.cpu_percent(None)
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    time.sleep(seconds)
    cpu = psutil.cpu_percent(None)
    stop.set()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    fps = [n / dt for n, _ in out.values()]
    return {"hw": hw, "streams": streams, "decoded_fps_per_stream_min": round(min(fps), 1),
            "decoded_fps_aggregate": round(sum(fps), 1),
            "converted_fps_aggregate": round(sum(c for _, c in out.values()) / dt, 1), "host_cpu_percent": cpu}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 4, 10])
    ap.add_argument("--hw", nargs="+", default=["none", "cuda"])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--convert-every", type=int, default=3, help="BGR-convert every k-th frame (pipeline: ~2.5)")
    ap.add_argument("--tag", default="4k")
    a = ap.parse_args()
    with av.open(a.clip) as c:
        s = c.streams.video[0]
        src = {"codec": s.codec_context.name, "width": s.width, "height": s.height, "fps": float(s.average_rate or 0)}
    rows = []
    for hw in a.hw:
        for n in a.streams:
            r = run(a.clip, n, hw, a.seconds, a.convert_every)
            print(json.dumps(r), flush=True)
            rows.append(r)
    out = {"source": src, "convert_every": a.convert_every, "seconds": a.seconds, "cpu": psutil.cpu_count(),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": "decode-only capacity: no inference, no RTSP",
           "rows": rows}
    dst = ROOT / "docs" / "benchmarks" / "results" / f"decode_{a.tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
