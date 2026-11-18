"""Teacher review: fusion, asymmetric verdict policy, resilience, provider wire formats, config defaults."""

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from app.config.settings import REPO_ROOT, AppConfig, TeacherConfig, _load_yaml
from app.inference.teacher import (
    QUESTION_ID,
    CircuitBreaker,
    FakeTeacher,
    GeminiDecisionTeacher,
    RateLimiter,
    SystemOneTeacher,
    agrees,
    decide,
    fuse,
)
from app.services.teacher_review import TeacherReviewer

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_fusion_math():
    assert fuse(1.0, 0.0, 0.7) == pytest.approx(0.7)
    assert fuse(0.2, 0.6, 0.25) == pytest.approx(0.25 * 0.2 + 0.75 * 0.6)
    assert fuse(0.9, 0.1, 0.0) == pytest.approx(0.1) and fuse(0.9, 0.1, 1.0) == pytest.approx(0.9)


def test_verdict_policy_thresholds():
    assert decide("violation", 0.19, 0.8, 0.2) == "LIKELY_FALSE_ALERT"
    assert decide("violation", 0.2, 0.8, 0.2) == "CONFIRMED"
    assert decide("compliant", 0.81, 0.8, 0.2) == "ESCALATED"
    assert decide("compliant", 0.8, 0.8, 0.2) == "COMPLIANT"
    assert agrees("violation", 0.7) and agrees("compliant", 0.1) and not agrees("compliant", 0.6)


def test_teacher_can_never_suppress_an_alert():
    """Over the whole probability / weight / threshold space a violation is only ever CONFIRMED or tagged for
    human review, and a compliant sample only ever stays compliant or is ESCALATED (adds an alert)."""
    grid = [i / 20 for i in range(21)]
    for p_t in grid:
        for p_m in grid:
            for w in grid:
                for esc, fa in ((0.8, 0.2), (0.5, 0.5), (0.0, 1.0), (1.0, 0.0)):
                    f = fuse(p_t, p_m, w)
                    assert decide("violation", f, esc, fa) in ("CONFIRMED", "LIKELY_FALSE_ALERT")
                    assert decide("compliant", f, esc, fa) in ("COMPLIANT", "ESCALATED")


def test_circuit_breaker_opens_and_half_opens():
    clk = Clock()
    b = CircuitBreaker(failures=3, cooldown_s=30, clock=clk)
    for _ in range(2):
        b.failure()
    assert b.allow() and b.state == "closed"
    b.failure()
    assert not b.allow() and b.state == "open" and b.retry_in() == pytest.approx(30)
    clk.t += 30
    assert b.allow() and b.state == "half_open"
    b.failure()  # failed trial -> open again for a full cooldown
    assert not b.allow()
    clk.t += 30
    b.success()
    assert b.state == "closed" and b.consecutive == 0


def test_rate_limiter_sliding_window():
    clk = Clock()
    r = RateLimiter(3, clock=clk)
    assert all(r.try_acquire() for _ in range(3))
    assert not r.try_acquire() and r.wait_time() == pytest.approx(60)
    clk.t += 59.9
    assert not r.try_acquire()
    clk.t += 0.1  # all three stamps are now 60 s old
    assert all(r.try_acquire() for _ in range(3)) and not r.try_acquire()


def _transport(seen, status=200, body=None):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(status, json=body or {})

    return httpx.MockTransport(handler)


async def test_gemini_request_and_response():
    seen = []
    t = GeminiDecisionTeacher(
        "test-key-123",
        transport=_transport(
            seen,
            body={
                "answers": {QUESTION_ID: {"type": "noul", "noul": 0.87, "confidence": 0.87}},
                "usage": {"total_tokens": 9},
            },
        ),
    )
    v = await t.review(JPEG)
    (req,) = seen
    assert str(req.url) == "https://generativelanguage.googleapis.com/v1beta/decisions" and req.method == "POST"
    assert req.headers["x-goog-api-key"] == "test-key-123" and "test-key-123" not in str(req.url)
    body = json.loads(req.content)
    assert body["model"] == "gemini-decision-1"
    assert body["input"] == [{"type": "image", "data": base64.b64encode(JPEG).decode(), "mime_type": "image/jpeg"}]
    assert body["questions"][QUESTION_ID]["type"] == "noul"
    assert v.p_no_helmet == 0.87 and v.provider == "gemini" and v.raw["usage"]["total_tokens"] == 9


async def test_gemini_errors_raise():
    with pytest.raises(httpx.HTTPStatusError) as e:
        await GeminiDecisionTeacher("k", transport=_transport([], 429, {"error": {"code": 429}})).review(JPEG)
    assert e.value.response.status_code == 429
    with pytest.raises(ValueError):
        await GeminiDecisionTeacher(
            "k", transport=_transport([], body={"answers": {QUESTION_ID: {"noul": 1.7}}})
        ).review(JPEG)
    with pytest.raises(KeyError):  # safety block / unexpected shape: no answers
        await GeminiDecisionTeacher("k", transport=_transport([], body={"usage": {}})).review(JPEG)


async def test_systemone_request_and_response():
    seen = []
    t = SystemOneTeacher(
        "http://clef.local:8080/", "clef-vision-1", transport=_transport(seen, body={"answers": {QUESTION_ID: 0.12}})
    )
    v = await t.review(JPEG)
    (req,) = seen
    assert str(req.url) == "http://clef.local:8080/v1/systemone" and "authorization" not in req.headers
    body = json.loads(req.content)
    assert body["model"] == "clef-vision-1" and isinstance(body["state"], str)
    assert body["images"][0]["data"] == base64.b64encode(JPEG).decode()
    assert body["questions"] == {
        QUESTION_ID: {"type": "noul", "instructions": body["questions"][QUESTION_ID]["instructions"]}
    }
    assert v.p_no_helmet == 0.12 and v.provider == "systemone"


def test_config_defaults_disabled():
    assert TeacherConfig().enabled is False
    cfg = AppConfig.model_validate(_load_yaml(REPO_ROOT / "configs", "app"))
    assert cfg.teacher.enabled is False and cfg.teacher.provider == "gemini"
    assert AppConfig().teacher.enabled is False
    with pytest.raises(ValueError):
        TeacherConfig(weight=1.5)


class _Evidence:
    def open(self, rel):
        return SimpleNamespace(read_bytes=lambda: JPEG)


def _cand(kind="violation"):
    return {
        "kind": kind,
        "event_id": None,
        "camera_id": "cam01",
        "track_id": 1,
        "rule_id": "r",
        "zone_id": None,
        "severity": "high",
        "person_box": [0, 0, 1, 1],
        "crop_path": "teacher/x.jpg",
        "p_model": 0.8,
        "captured_at": 0.0,
    }


async def test_reviewer_queue_drops_oldest():
    r = TeacherReviewer(TeacherConfig(), FakeTeacher(), None, _Evidence(), ["dashboard"], maxsize=2)
    for i in range(5):
        r._enqueue({**_cand(), "track_id": i})
    assert [c["track_id"] for c in r.q] == [3, 4] and r.dropped == 3


async def test_reviewer_failures_and_timeouts_are_harmless():
    r = TeacherReviewer(TeacherConfig(timeout_s=0.05), FakeTeacher(fail=True), None, _Evidence(), ["dashboard"])
    for _ in range(5):
        assert await r.review_one(_cand()) is None  # nothing persisted (sf=None would raise if touched)
    assert r.failures == 5 and r.breaker.state == "open" and "ConnectError" in r.health()["last_error"]
    slow = TeacherReviewer(TeacherConfig(timeout_s=0.05), FakeTeacher(delay_s=1.0), None, _Evidence(), [])
    t0 = asyncio.get_running_loop().time()
    assert await slow.review_one(_cand()) is None and asyncio.get_running_loop().time() - t0 < 0.5
    assert slow.failures == 1
