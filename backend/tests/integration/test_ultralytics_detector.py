"""GPU test: real cascade on a real demo frame (needs models/ and the demo clips)."""

import time

import av
import pytest

from app.config.settings import REPO_ROOT, load_models_config
from app.streaming.frame import Frame

pytestmark = pytest.mark.gpu


def first_frames(path, n=1, skip=30):
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
    return (frames[skip:] or frames)[:n]  # short demo clips: skip what is not there


def test_cascade_on_demo_frames(streams_dir):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if not (REPO_ROOT / "models" / "ppe_yolo26s.pt").exists():
        pytest.skip("models/ppe_yolo26s.pt missing")
    from app.inference.ultralytics_detector import UltralyticsCascadeDetector

    det = UltralyticsCascadeDetector(load_models_config(), REPO_ROOT / "models")
    det.warmup()
    imgs = first_frames(streams_dir / "cam07.mp4") + first_frames(streams_dir / "cam05.mp4")
    frames = [Frame(f"cam{i}", 1, im, None, time.time(), time.monotonic()) for i, im in enumerate(imgs)]
    out = det.detect(frames)
    assert len(out) == 2
    persons = [d for fd in out for d in fd.detections if d.cls == "person"]
    assert persons and all(p.keypoints is not None and p.keypoints.shape == (17, 3) for p in persons)
    ppe = [d for fd in out for d in fd.detections if d.cls in ("helmet", "head")]
    assert ppe, "expected helmet/head detections on demo frames"
    expected = {"person", "ppe"} | ({"items"} if det.items else set()) | ({"phone"} if det.phone else set())
    assert "@" in det.version and set(det.model_versions()) == expected
