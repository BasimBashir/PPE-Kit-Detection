import numpy as np

from app.config.settings import TrackingConfig
from app.inference.mock_detector import MockDetector, person_with_helmet, person_without_helmet
from app.inference.types import Detection
from app.streaming.frame import Frame
from app.tracking.tracker import make_tracker


def fr(i, cam="cam01"):
    return Frame(cam, i, np.zeros((720, 1280, 3), np.uint8), None, float(i), float(i))


def person(x, y=100, conf=0.9, kps=None):
    return Detection("person", conf, (x, y, x + 80, y + 240), "t", kps)


def test_stable_id_for_smooth_motion_and_keypoints_preserved():
    t = make_tracker(TrackingConfig(), frame_rate=5)
    kps = np.ones((17, 3), np.float32)
    ids = set()
    for i in range(20):
        out = t.update([person(100 + 6 * i, kps=kps)], fr(i))
        if out:
            ids.add(out[0].track_id)
            assert out[0].keypoints is kps
    assert len(ids) == 1 and t.active_count == 1


def test_two_trackers_are_independent():
    a, b = make_tracker(TrackingConfig(), 5), make_tracker(TrackingConfig(), 5)
    for i in range(5):
        oa = a.update([person(100 + 5 * i), person(600 + 5 * i)], fr(i, "cam01"))
        ob = b.update([person(300 + 5 * i)], fr(i, "cam02"))
    assert len(oa) == 2 and len(ob) == 1
    # each tracker numbers its own tracks from 1: cam02's track does not continue cam01's counter
    assert min(x.track_id for x in ob) == 1 and max(x.track_id for x in oa) == 2


def test_reset_and_empty_update():
    t = make_tracker(TrackingConfig(), 5)
    t.update([person(100)], fr(0))
    t.update([person(102)], fr(1))
    assert t.update([], fr(2)) == []
    t.reset()
    out = t.update([person(500)], fr(3))
    assert all(o.track_id == 1 for o in out)


def test_mock_detector_is_deterministic():
    d = MockDetector()
    f1, f2 = fr(7, "cam03"), fr(7, "cam03")
    assert [x.xyxy for x in d.detect([f1])[0].detections] == [x.xyxy for x in d.detect([f2])[0].detections]
    cls_odd = {x.cls for x in d.detect([fr(1, "cam03")])[0].detections}
    cls_even = {x.cls for x in d.detect([fr(1, "cam04")])[0].detections}
    assert cls_odd == {"person", "head"} and cls_even == {"person", "helmet"}
    assert {x.cls for x in person_with_helmet(0, 0, 100, 300)} == {"person", "helmet"}
    assert {x.cls for x in person_without_helmet(0, 0, 100, 300)} == {"person", "head"}
