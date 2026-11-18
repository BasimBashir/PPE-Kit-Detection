"""Teacher review: an external vision decision model re-checks head crops AFTER the on-site decision.

Never on the alert path: reviews run asynchronously in the API process (app/services/teacher_review.py), so a slow
or unreachable teacher can never delay or block an alert. Asymmetric safety rule: the teacher may ESCALATE a
compliant sample or tag an alert LIKELY_FALSE_ALERT for human review; it can never suppress or change an alert.
"""

from __future__ import annotations

import asyncio
import base64
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

QUESTION_ID = "no_helmet"
INSTRUCTIONS = (
    "This is a crop of one person's head from a construction-site safety camera. Is the person NOT "
    "wearing a hard hat / safety helmet (bare head, cap, hood or hair only)?"
)
STATE = "Head crop from a construction-site PPE camera; decide whether a hard hat is missing."

Kind = Literal["violation", "compliant"]
Verdict = Literal["CONFIRMED", "LIKELY_FALSE_ALERT", "ESCALATED", "COMPLIANT"]


@dataclass
class TeacherVerdict:
    p_no_helmet: float  # probability that the person is NOT wearing a hard hat
    latency_ms: float
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


class Teacher(Protocol):
    provider: str

    async def review(self, crop_jpeg: bytes) -> TeacherVerdict: ...


def _prob(v: Any) -> float:
    p = float(v)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"teacher probability out of range: {p}")
    return p


class _HttpTeacher:
    provider = "http"

    def __init__(
        self, url: str, headers: dict[str, str], timeout_s: float, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.url, self.headers, self.timeout_s, self.transport = url, headers, timeout_s, transport

    def build_body(self, crop_jpeg: bytes) -> dict[str, Any]:
        raise NotImplementedError

    def parse(self, data: dict[str, Any]) -> float:
        raise NotImplementedError

    async def review(self, crop_jpeg: bytes) -> TeacherVerdict:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout_s) as c:
            r = await c.post(self.url, json=self.build_body(crop_jpeg), headers=self.headers)
        r.raise_for_status()  # HTTPStatusError carries the status (429 => back off); the API key is in a header only
        data = r.json()
        return TeacherVerdict(self.parse(data), 1000 * (time.perf_counter() - t0), self.provider, data)


class GeminiDecisionTeacher(_HttpTeacher):
    """Google Gemini Decisions API (gemini-decision-1): POST {base}/decisions, `noul` question => P(yes)."""

    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-decision-1",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(f"{base_url.rstrip('/')}/decisions", {"x-goog-api-key": api_key}, timeout_s, transport)
        self.model = model

    def build_body(self, crop_jpeg: bytes) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [{"type": "image", "data": base64.b64encode(crop_jpeg).decode(), "mime_type": "image/jpeg"}],
            "questions": {QUESTION_ID: {"type": "noul", "instructions": INSTRUCTIONS}},
        }

    def parse(self, data: dict[str, Any]) -> float:
        return _prob(data["answers"][QUESTION_ID]["noul"])


class SystemOneTeacher(_HttpTeacher):
    """Generic SystemOne wire protocol (Jev / self-hosted Cloudflare Clef): POST {base}/v1/systemone.
    shortcut: coded against the published request/response shape only, not run live; verify on first deployment."""

    provider = "systemone"

    def __init__(
        self, base_url: str, model: str, timeout_s: float = 5.0, transport: httpx.AsyncBaseTransport | None = None
    ):
        super().__init__(f"{base_url.rstrip('/')}/v1/systemone", {}, timeout_s, transport)
        self.model = model

    def build_body(self, crop_jpeg: bytes) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": STATE,
            "images": [{"data": base64.b64encode(crop_jpeg).decode(), "mime_type": "image/jpeg"}],
            "questions": {QUESTION_ID: {"type": "noul", "instructions": INSTRUCTIONS}},
        }

    def parse(self, data: dict[str, Any]) -> float:
        return _prob(data["answers"][QUESTION_ID])


class FakeTeacher:
    """Deterministic teacher for tests: fixed probability or a function of the crop bytes."""

    provider = "fake"

    def __init__(self, p: float | Callable[[bytes], float] = 0.5, fail: bool = False, delay_s: float = 0.0):
        self.p, self.fail, self.delay_s, self.calls = p, fail, delay_s, 0

    async def review(self, crop_jpeg: bytes) -> TeacherVerdict:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise httpx.ConnectError("fake teacher unavailable")
        p = self.p(crop_jpeg) if callable(self.p) else self.p
        return TeacherVerdict(p, self.delay_s * 1000, self.provider, {"p": p})


# ---------------------------------------------------------------- fusion + asymmetric policy


def fuse(p_teacher: float, p_model: float, w: float) -> float:
    """fused = w * p_teacher + (1 - w) * p_model; all probabilities are P(no helmet)."""
    return w * p_teacher + (1.0 - w) * p_model


def decide(kind: Kind, fused: float, escalate_threshold: float, false_alert_threshold: float) -> Verdict:
    """The teacher can only ADD signal: escalate a compliant sample, or tag an alert for human review.
    No verdict maps to suppressing, resolving, acknowledging or changing an alert - by construction."""
    if kind == "violation":
        return "LIKELY_FALSE_ALERT" if fused < false_alert_threshold else "CONFIRMED"
    return "ESCALATED" if fused > escalate_threshold else "COMPLIANT"


def agrees(kind: Kind, p_teacher: float) -> bool:
    """Teacher alone (p >= 0.5 => no helmet) agrees with the on-site decision."""
    return (p_teacher >= 0.5) == (kind == "violation")


# ---------------------------------------------------------------- resilience


class CircuitBreaker:
    """Open after `failures` consecutive failures; allow one trial call after `cooldown_s` (half-open)."""

    def __init__(self, failures: int = 5, cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self.clock = failures, cooldown_s, clock
        self.consecutive = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.cooldown_s else "open"

    def retry_in(self) -> float:
        return 0.0 if self.opened_at is None else max(0.0, self.opened_at + self.cooldown_s - self.clock())

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self.consecutive, self.opened_at = 0, None

    def failure(self) -> None:
        self.consecutive += 1
        if self.consecutive >= self.threshold:
            self.opened_at = self.clock()  # a failed half-open trial re-opens for another cooldown


class RateLimiter:
    """At most `per_minute` acquisitions in any sliding 60 s window."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self.stamps: deque[float] = deque()

    def wait_time(self) -> float:
        now = self.clock()
        while self.stamps and now - self.stamps[0] >= 60.0:
            self.stamps.popleft()
        return 0.0 if len(self.stamps) < self.per_minute else 60.0 - (now - self.stamps[0])

    def try_acquire(self) -> bool:
        if self.wait_time() > 0:
            return False
        self.stamps.append(self.clock())
        return True


def build_teacher(cfg: Any, api_key: str | None) -> Teacher:
    """cfg: TeacherConfig. The key is only ever passed in a request header, never logged."""
    if cfg.provider == "fake":
        return FakeTeacher(lambda b: random.Random(b).random())  # deterministic per crop (demo / tests)
    if cfg.provider == "systemone":
        return SystemOneTeacher(cfg.base_url, cfg.model, cfg.timeout_s)  # self-hosted: the Gemini key never goes there
    if not api_key:
        raise ValueError("teacher.provider=gemini needs GEMINI_API_KEY in the environment / .env")
    return GeminiDecisionTeacher(api_key, cfg.model, cfg.base_url, cfg.timeout_s)
