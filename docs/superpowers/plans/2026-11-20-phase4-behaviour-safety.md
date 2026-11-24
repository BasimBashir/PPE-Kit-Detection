# Phase 4 (0.4.0) — Behaviour & Safety Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four per-person checks — **fall** (confirmed by a temporal sequence), **fighting**, **phone use during work hours** and **stalling work** — enforced through the existing rule → state machine → event → alert path, configurable from the UI, and measured on public data.

**Architecture:** A behaviour is a new kind of *check* a rule can enforce, next to PPE items. Each behaviour has an *assessor* with the same interface as `Associator` (`assess_deferred`, `take_pending`, `set_thresholds`) that emits per-track `Observation`s: `ABSENT` = the behaviour is happening (violation evidence), `PRESENT` = normal, `UNCERTAIN` = not assessable. Everything downstream — `PPEStateMachine` (N-of-M / duration confirmation, hysteresis), `EventService` (idempotency, cooldown, spatial dedupe, outbox, spool), evidence, presets, the Events page — is reused unchanged. A per-rule **shift schedule** (`schedule` JSONB column) gates any rule by site-local time. Fall and stalling are rule-based on the pose keypoints the cascade already produces; phone adds an optional COCO detector stage; fighting adds a tiny GRU over pair keypoint sequences, trained by a script that shares the production feature code.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2 + Alembic, pydantic 2, Ultralytics YOLO26 (pose + COCO detect), PyTorch (GRU), React + TypeScript + Vitest.

## Global Constraints

- Scope (user decision 2026-11-20): fall (temporal confirmation), fighting, phone use in work hours, stalling work, normal (= no event). **Dropped:** angry, not working. **Parked:** accident detection (until a client proposal).
- Deployment region: Gulf (UAE / KSA). Privacy notice / consent is a deployment checklist item (docs only), not code.
- Every operator setting must be editable in the UI (rules, hold times, schedule, zones). Engineering constants live in `configs/app.yaml`.
- No new runtime dependency except `tzdata` (explicit; zoneinfo needs it on Windows). Dataset helpers (`omnifall`, `datasets`) run only via `uv run --with ...`, never added to `pyproject.toml`.
- No paid APIs. The teacher review layer stays PPE-only (it reviews single crops; behaviours need temporal context).
- Never claim an accuracy we did not measure. Stalling gets timer-correctness tests only — no accuracy figure exists.
- Datasets are evaluation/MVP only: OmniFall annotations CC BY-NC-SA 4.0; RLVS "© original authors"; SCFD repo MIT (videos from YouTube); RWF-2000 non-commercial and currently unavailable. Fight weights trained here must be retrained on licensed/client data before commercial use — record this in `docs/model-and-data-licenses.md`.
- Downloaded datasets are untrusted: keep them under `data/` (one new directory each), never run code from inside them.
- Python: ruff line-length 120, rules `E,F,W,I,B,UP,SIM,S`. Tests: `cd backend && uv run pytest tests -q -m "not gpu"` (venv: `UV_PROJECT_ENVIRONMENT=C:/Users/mianx/.venvs/ppe-intelligence`, see docs/implementation-plan.md §1). Integration tests need the Docker Postgres up.
- Frontend: `cd frontend && npm run lint && npx vitest run && npm run build`.
- Commit after each task on the current branch; message style `feat(phase4): ...` / `docs(phase4): ...`.
- Keep the field name `ppe_class` (DB column + API) as the check key — renaming needs a migration and breaks clients; comment it instead.

## File map

| File | Responsibility |
|---|---|
| `backend/app/config/settings.py` | `Behaviour` catalogue, `Schedule`, `BehaviourConfig`, `check_meta()`, `RuleSeed.schedule` + behaviour defaults, `ModelsConfig.phone/fight_weights` |
| `backend/alembic/versions/0004_rule_schedule.py` | `ppe_rules.schedule` JSONB column |
| `models/tables.py`, `schemas/serializers.py`, `schemas/requests.py`, `api/routes_events.py`, `repositories/rules.py`, `alerts/event_service.py` | persist/expose schedule; catalogue endpoint returns `kind`; behaviour labels; teacher guard |
| `backend/app/behaviour/__init__.py` | `make_behaviour_assessor()` factory |
| `backend/app/behaviour/assessors.py` | shared gate/history plumbing, `torso()`, `FallAssessor`, `IdleAssessor`, `PhoneAssessor` |
| `backend/app/behaviour/fight.py` | `pair_features`, `PairSequencer`, `FightGRU`, `FightModel`, `FightAssessor` |
| `backend/app/inference/ultralytics_detector.py` | optional `phone` stage |
| `backend/app/workers/pipeline_worker.py` | assessor selection, schedule gating, fight model load |
| `frontend/src/...` | check select (PPE + behaviour), `ScheduleEditor`, labels |
| `scripts/ml/omnifall_to_csv.py`, `eval_fall.py`, `extract_fight_sequences.py`, `train_fight.py`, `eval_phone_coco.py` | measurement |
| `scripts/benchmark/run_benchmark.py` | `p4_*` profile (GPU cost of phone stage + behaviour rules) |
| `docs/decisions/ADR-015-behaviour-checks.md`, `docs/ml/phase4-log.md`, README, CHANGELOG, licences, privacy | documentation |

---

### Task 1: Check catalogue, shift schedule and behaviour config

**Files:**
- Modify: `backend/app/config/settings.py`
- Modify: `backend/pyproject.toml` (dependencies: add `"tzdata>=2025.2",`)
- Modify: `configs/app.example.yaml`, `configs/models.example.yaml` (commented examples)
- Test: `backend/tests/unit/test_behaviour_config.py`

**Interfaces:**
- Produces: `Behaviour(label, alert, policy: TemporalPolicy, thresholds: RuleThresholds)`, `BEHAVIOURS: dict[str, Behaviour]` with keys `fall`, `fight`, `phone`, `stalling`; `check_meta(key) -> PPEItem | Behaviour`; `Schedule(tz: str, days: list[int], windows: list[str])` with `.active(wall: float) -> bool`; `RuleSeed.schedule: Schedule | None`; `BehaviourConfig` as `AppConfig.behaviour`; `ModelsConfig.phone: ModelSpec | None`, `ModelsConfig.phone_class_map = {"phone": 67}`, `ModelsConfig.fight_weights: str | None`.

- [ ] **Step 1: Write the failing tests** — create `backend/tests/unit/test_behaviour_config.py`:

```python
"""Phase 4: check catalogue (PPE items + behaviours), behaviour rule defaults, shift schedules."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app.config.settings import BEHAVIOURS, AppConfig, ModelsConfig, RuleSeed, Schedule, check_meta

DXB = ZoneInfo("Asia/Dubai")


def at(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=DXB).timestamp()


def test_behaviour_rules_validate_and_get_tuned_defaults():
    r = RuleSeed(id="fall-yard", name="Fall", ppe_class="fall")
    assert r.temporal_policy == BEHAVIOURS["fall"].policy and r.temporal_policy.mode == "duration"
    assert RuleSeed(id="f", name="F", ppe_class="fight").thresholds.absent_conf == 0.7
    assert RuleSeed(id="s", name="S", ppe_class="stalling").temporal_policy.min_duration_s == 600
    custom = RuleSeed(id="p", name="P", ppe_class="phone", temporal_policy={"mode": "count"})
    assert custom.temporal_policy.mode == "count"  # an explicit policy is never overwritten
    with pytest.raises(ValidationError):
        RuleSeed(id="x", name="X", ppe_class="angry")


def test_check_meta_covers_items_and_behaviours():
    assert check_meta("hard_hat").alert == "NO HARD HAT" and check_meta("fall").alert == "PERSON DOWN"
    assert set(BEHAVIOURS) == {"fall", "fight", "phone", "stalling"}


def test_schedule_day_windows_and_breaks():
    s = Schedule(tz="Asia/Dubai", days=[0, 1, 2, 3, 4], windows=["07:00-12:00", "13:00-17:00"])
    assert datetime(2026, 11, 23, tzinfo=DXB).weekday() == 0  # Monday
    assert s.active(at(2026, 11, 23, 7, 0)) and s.active(at(2026, 11, 23, 11, 59))
    assert not s.active(at(2026, 11, 23, 12, 30))  # lunch break
    assert not s.active(at(2026, 11, 23, 17, 0))  # end is exclusive
    assert not s.active(at(2026, 11, 22, 8, 0))  # Sunday is not a work day here


def test_overnight_window_belongs_to_its_start_day():
    riy = ZoneInfo("Asia/Riyadh")
    s = Schedule(tz="Asia/Riyadh", days=[0], windows=["22:00-06:00"])
    assert s.active(datetime(2026, 11, 23, 23, 0, tzinfo=riy).timestamp())  # Monday night
    assert s.active(datetime(2026, 11, 24, 5, 0, tzinfo=riy).timestamp())  # Tuesday early = Monday's shift
    assert not s.active(datetime(2026, 11, 23, 5, 0, tzinfo=riy).timestamp())  # Monday early = Sunday's shift


@pytest.mark.parametrize("bad", [{"tz": "Mars/Base"}, {"days": []}, {"days": [7]}, {"days": [1, 1]},
                                 {"windows": ["7:00-12:00"]}, {"windows": ["12:00-12:00"]}, {"windows": []}])
def test_schedule_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        Schedule(**{"tz": "Asia/Dubai", "days": [0], "windows": ["07:00-12:00"], **bad})


def test_rule_schedule_and_model_config_defaults():
    r = RuleSeed(id="p", name="P", ppe_class="phone",
                 schedule={"tz": "Asia/Dubai", "days": [0], "windows": ["07:00-12:00"]})
    assert isinstance(r.schedule, Schedule) and RuleSeed(id="q", name="Q").schedule is None
    m = ModelsConfig.model_validate({"person": {"weights": "p.pt"}, "ppe": {"weights": "h.pt"}})
    assert m.phone is None and m.fight_weights is None and m.phone_class_map == {"phone": 67}
    assert AppConfig().behaviour.lying_min_deg > AppConfig().behaviour.upright_max_deg
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && uv run pytest tests/unit/test_behaviour_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'BEHAVIOURS'`.

- [ ] **Step 3: Implement in `backend/app/config/settings.py`**

3a. Imports: change `from typing import Literal` to `from typing import Any, Literal`, and add with the stdlib imports:

```python
import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
```

3b. Add after `TeacherConfig`, and add `behaviour: BehaviourConfig = BehaviourConfig()` as the last field of `AppConfig`:

```python
class BehaviourConfig(BaseModel):
    """Phase 4 behaviour checks (ADR-015). Engineering constants; the operator knobs (hold time = the rule's
    min_duration_s, shift schedule, evidence confidence = the rule's absent_conf) live on each rule (UI)."""

    keypoint_min_conf: float = Field(0.4, ge=0, le=1)
    # fall: torso upright -> lying within fall_window_s while the hips drop >= drop_frac of the standing height
    fall_window_s: float = Field(2.0, gt=0)
    upright_max_deg: float = Field(35.0, ge=0, le=90)  # torso angle from vertical
    lying_min_deg: float = Field(60.0, ge=0, le=90)
    drop_frac: float = Field(0.25, gt=0, le=1)
    # stalling: every keypoint stays within idle_max_range body heights (5th-95th percentile) over idle_window_s
    idle_window_s: float = Field(10.0, gt=0)
    idle_max_range: float = Field(0.08, gt=0)
    # phone: a phone box over a (grown) hand or head ROI; detector dropouts shorter than phone_gap_s are bridged
    phone_roi_scale: float = Field(1.8, ge=1)
    phone_min_ioa: float = Field(0.3, gt=0, le=1)
    phone_gap_s: float = Field(2.0, ge=0)
    # fight: two people closer than fight_pair_dist mean body heights form a pair scored by the fight model
    fight_pair_dist: float = Field(1.2, gt=0)
```

3c. In `ModelsConfig`, after `cascade_skip_ppe_without_person: bool = True`:

```python
    # Phase 4 (ADR-015): COCO detector for the phone check ("cell phone" = COCO class 67); None = phone rules
    # stay UNCERTAIN. fight_weights: models/<file> from scripts/ml/train_fight.py; None = fight rules UNCERTAIN.
    phone: ModelSpec | None = None
    phone_class_map: dict[str, int] = {"phone": 67}
    fight_weights: str | None = None
```

3d. Directly after `class RuleThresholds` (before `class RuleSeed`):

```python
class Behaviour(BaseModel):
    """Phase 4 (ADR-015): a per-person behaviour a rule can enforce, like a PPE item. Its assessor emits ABSENT =
    the behaviour is happening (violation evidence), PRESENT = normal, UNCERTAIN = not assessable, so the PPE state
    machine, events, dedupe and alerts apply unchanged."""

    label: str
    alert: str
    policy: TemporalPolicy  # default temporal policy for new rules of this check (editable per rule)
    thresholds: RuleThresholds = RuleThresholds()


BEHAVIOURS: dict[str, Behaviour] = {
    # "down after a fall" held min_duration_s; the fall itself is the temporal sequence in FallAssessor
    "fall": Behaviour(label="Fall", alert="PERSON DOWN", policy=TemporalPolicy(
        mode="duration", min_duration_s=3.0, suspect_n=1, confirm_n=3, confirm_m=5, compliant_n=3, compliant_m=5,
        clear_n=5, clear_m=8, unknown_ratio=0.8, window=30)),
    "fight": Behaviour(label="Fighting", alert="FIGHT", thresholds=RuleThresholds(absent_conf=0.7),
                       policy=TemporalPolicy(mode="count", confirm_n=3, confirm_m=5, suspect_n=2, compliant_n=3,
                                             compliant_m=5, clear_n=6, clear_m=8, window=30)),
    "phone": Behaviour(label="Phone use", alert="PHONE USE", policy=TemporalPolicy(
        mode="duration", min_duration_s=10.0, suspect_n=2, compliant_n=4, compliant_m=6, clear_n=6, clear_m=8,
        unknown_ratio=0.8, window=30)),
    # measures stillness, not intent: scope it with a work-area zone and a shift schedule
    "stalling": Behaviour(label="Stalling work", alert="STALLING", policy=TemporalPolicy(
        mode="duration", min_duration_s=600.0, suspect_n=3, compliant_n=5, compliant_m=8, clear_n=5, clear_m=8,
        unknown_ratio=0.8, window=30)),
}


def check_meta(key: str) -> PPEItem | Behaviour:
    """A rule's check - PPE item or behaviour; both carry .label and .alert."""
    return PPE_ITEMS[key] if key in PPE_ITEMS else BEHAVIOURS[key]


_WINDOW = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


def _minutes(window: str) -> tuple[int, int]:
    h1, m1, h2, m2 = map(int, _WINDOW.match(window).groups())  # type: ignore[union-attr]
    return h1 * 60 + m1, h2 * 60 + m2


class Schedule(BaseModel):
    """When a rule is enforced (Phase 4): shift windows in site-local time, breaks left out. A window whose end is
    before its start runs past midnight and belongs to the day it starts. End is exclusive."""

    tz: str
    days: list[int] = Field(min_length=1, max_length=7)  # 0 = Monday ... 6 = Sunday
    windows: list[str] = Field(min_length=1, max_length=12)  # "HH:MM-HH:MM"

    @field_validator("tz")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"unknown time zone {v!r} (IANA name, e.g. Asia/Dubai)") from e
        return v

    @field_validator("days")
    @classmethod
    def _days(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v) or any(not 0 <= d <= 6 for d in v):
            raise ValueError("days: unique integers 0 (Monday) .. 6 (Sunday)")
        return v

    @field_validator("windows")
    @classmethod
    def _windows(cls, v: list[str]) -> list[str]:
        for w in v:
            if not _WINDOW.match(w) or _minutes(w)[0] == _minutes(w)[1]:
                raise ValueError(f"window {w!r}: use HH:MM-HH:MM with start != end")
        return v

    def active(self, wall: float) -> bool:
        now = datetime.fromtimestamp(wall, ZoneInfo(self.tz))
        t, day = now.hour * 60 + now.minute, now.weekday()
        for w in self.windows:
            s, e = _minutes(w)
            if s < e and day in self.days and s <= t < e:
                return True
            if s > e and ((day in self.days and t >= s) or ((day - 1) % 7 in self.days and t < e)):
                return True
        return False
```

3e. In `RuleSeed`: replace the `ppe_class` line with
`ppe_class: str = "hard_hat"  # the check: a PPE item (PPE_ITEMS) or a behaviour (BEHAVIOURS); name kept for API/DB`,
add `schedule: Schedule | None = None  # None = always enforced (Phase 4: shift hours, breaks left out)` after
`temporal_policy`, and replace the `_known_item` validator with:

```python
    @model_validator(mode="before")
    @classmethod
    def _behaviour_defaults(cls, data: Any) -> Any:
        """New behaviour rules start from that behaviour's tuned policy/thresholds, not the PPE defaults."""
        if isinstance(data, dict) and data.get("ppe_class") in BEHAVIOURS:
            b = BEHAVIOURS[data["ppe_class"]]
            data = {**data}
            if data.get("temporal_policy") is None:
                data["temporal_policy"] = b.policy.model_dump()
            if data.get("thresholds") is None:
                data["thresholds"] = b.thresholds.model_dump()
        return data

    @field_validator("ppe_class")
    @classmethod
    def _known_check(cls, v: str) -> str:
        if v not in PPE_ITEMS and v not in BEHAVIOURS:
            raise ValueError(f"unknown check {v!r}; configured: {', '.join([*PPE_ITEMS, *BEHAVIOURS])}")
        return v
```

3f. `backend/pyproject.toml` dependencies: add `"tzdata>=2025.2",  # zoneinfo data for shift schedules (Windows has none)`; run `cd backend && uv lock && uv sync`.

3g. Append to `configs/app.example.yaml`:

```yaml
# Phase 4 behaviour checks (ADR-015) - engineering constants; operator knobs are on each rule in the UI.
# behaviour:
#   fall_window_s: 2.0      # upright -> lying faster than this = a fall (slower = lying down on purpose)
#   drop_frac: 0.25         # hips must drop this share of the standing height
#   idle_window_s: 10.0     # stalling: stillness measured over this window
#   idle_max_range: 0.08    # ... every keypoint within this many body heights
#   fight_pair_dist: 1.2    # people closer than this (body heights) are scored as a pair
```

Append to `configs/models.example.yaml`:

```yaml
# Phase 4 (ADR-015). phone: COCO detector ("cell phone" = class 67); unset = phone rules stay UNCERTAIN.
# phone:
#   weights: yolo26s.pt
#   imgsz: 1280
#   conf: 0.25
#   half: true
#   device: cuda:0
# fight_weights: fight_gru.pt   # scripts/ml/train_fight.py; MVP weights - see docs/model-and-data-licenses.md
```

- [ ] **Step 4: Run tests** — `cd backend && uv run pytest tests/unit/test_behaviour_config.py tests/unit/test_multi_ppe.py tests/unit/test_ppe_logic.py -q` → PASS; `uv run ruff check app tests` → clean.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/settings.py backend/tests/unit/test_behaviour_config.py backend/pyproject.toml backend/uv.lock configs/app.example.yaml configs/models.example.yaml
git commit -m "feat(phase4): behaviour check catalogue, shift schedules, behaviour config"
```

---

### Task 2: Persist and expose schedules and behaviour checks (DB, API, events)

**Files:**
- Create: `backend/alembic/versions/0004_rule_schedule.py`
- Modify: `backend/app/models/tables.py` (`PPERule`), `backend/app/schemas/serializers.py` (`rule_out`), `backend/app/schemas/requests.py` (`RulePatch`, `PresetIn`), `backend/app/api/routes_events.py` (`ppe_items`), `backend/app/repositories/rules.py` (`apply_preset`), `backend/app/alerts/event_service.py` (`_create`)
- Test: `backend/tests/integration/test_api.py` (append)

**Interfaces:**
- Consumes: `BEHAVIOURS`, `Schedule`, `check_meta` (Task 1).
- Produces: `GET /api/v1/ppe-items` items `{id, label, region | null, kind: "ppe" | "behaviour"}`; rule JSON gains `schedule: {tz, days, windows} | null`; `PATCH /rules/{id}` accepts `schedule` (null clears it).

- [ ] **Step 1: Write the failing integration test** — append to `backend/tests/integration/test_api.py`:

```python
def test_behaviour_rules_defaults_schedule_and_catalogue(client):
    admin = login(client)
    checks = {i["id"]: i["kind"] for i in client.get("/api/v1/ppe-items", headers=admin).json()}
    assert checks["fall"] == "behaviour" and checks["stalling"] == "behaviour" and checks["hard_hat"] == "ppe"
    r = client.post("/api/v1/rules", headers=admin, json={"id": "fall-yard", "name": "Fall", "ppe_class": "fall"})
    assert r.status_code == 200
    body = r.json()
    assert body["temporal_policy"]["mode"] == "duration" and body["effective_confirm_seconds"] == 3.0
    assert body["schedule"] is None
    sched = {"tz": "Asia/Riyadh", "days": [6, 0, 1, 2, 3], "windows": ["07:00-12:00", "22:00-06:00"]}
    ok = client.patch("/api/v1/rules/fall-yard", headers=admin, json={"schedule": sched})
    assert ok.status_code == 200 and ok.json()["schedule"] == sched
    listed = {x["id"]: x for x in client.get("/api/v1/rules", headers=admin).json()}
    assert listed["fall-yard"]["schedule"] == sched  # survives the DB round trip
    bad = client.patch("/api/v1/rules/fall-yard", headers=admin, json={"schedule": {**sched, "tz": "Mars/Base"}})
    assert bad.status_code == 422
    assert client.patch("/api/v1/rules/fall-yard", headers=admin, json={"schedule": None}).json()["schedule"] is None
    assert client.post("/api/v1/rules", headers=admin,
                       json={"id": "x", "name": "X", "ppe_class": "angry"}).status_code == 422
    put = client.put("/api/v1/presets/gulf-yard", headers=admin,
                     json={"items": {"hard_hat": "mandatory", "phone": "secondary"}})
    assert put.status_code == 200
    applied = client.post("/api/v1/presets/gulf-yard/apply", headers=admin, json={"camera_id": "cam01"}).json()
    assert {a["ppe_class"] for a in applied} == {"hard_hat", "phone"}
    assert next(a for a in applied if a["ppe_class"] == "phone")["temporal_policy"]["min_duration_s"] == 10.0
```

- [ ] **Step 2: Run to verify failure** — `cd backend && uv run pytest tests/integration/test_api.py -k behaviour -q` → FAIL (`KeyError: 'kind'`).

- [ ] **Step 3: Migration** — create `backend/alembic/versions/0004_rule_schedule.py`:

```python
"""rule shift schedule (Phase 4): ppe_rules.schedule JSONB, NULL = always enforced

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-20 18:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ppe_rules", sa.Column("schedule", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("ppe_rules", "schedule")
```

- [ ] **Step 4: ORM + serializer + requests + routes + repo + events**

`backend/app/models/tables.py`, in `PPERule` after `temporal_policy`:

```python
    schedule: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # Phase 4 shift schedule; NULL = always
```

`backend/app/schemas/serializers.py` `rule_out`: change `"temporal_policy": tp,` to `"temporal_policy": tp, "schedule": r.schedule,`.

`backend/app/schemas/requests.py`: import line becomes
`from app.config.settings import BEHAVIOURS, PPE_ITEMS, RuleThresholds, Schedule, TemporalPolicy`;
in `RulePatch` add `schedule: Schedule | None = None  # explicit null clears it (exclude_unset in the route)`;
in `PresetIn._known` replace the body with:

```python
        known = set(PPE_ITEMS) | set(BEHAVIOURS)
        unknown = set(v) - known
        if unknown or not v:
            raise ValueError(f"items must be non-empty and from {sorted(known)}; unknown: {sorted(unknown)}")
        return v
```

`backend/app/api/routes_events.py`: import `BEHAVIOURS` with `PPE_ITEMS`; replace `ppe_items` body:

```python
    """The check catalogue: PPE items (models.yaml ppe_items) and Phase 4 behaviours - what rules can enforce."""
    return ([{"id": k, "label": v.label, "region": v.region, "kind": "ppe"} for k, v in PPE_ITEMS.items()]
            + [{"id": k, "label": v.label, "region": None, "kind": "behaviour"} for k, v in BEHAVIOURS.items()])
```

`backend/app/repositories/rules.py`: import `check_meta` instead of `PPE_ITEMS`
(`from app.config.settings import RuleSeed, ZoneSeed, check_meta`) and in `apply_preset` use
`name=f"{check_meta(item).label} ({name})"`. (`upsert_rule` needs no change: `seed.model_dump()` already contains
`schedule` as a dict or None; `rule_to_seed` reads every `RuleSeed` field incl. `schedule`.)

`backend/app/alerts/event_service.py`: import `from app.config.settings import BEHAVIOURS, AlertsConfig`; in
`_create` replace the `alert = ...` line with

```python
        meta = ITEMS.get(v.ppe_class) or BEHAVIOURS.get(v.ppe_class)
        alert = meta.alert if meta else v.ppe_class
```

and change the teacher condition to
`if self.teacher_enabled and roi is not None and v.ppe_class in ITEMS:  # teacher reviews PPE crops only`.

- [ ] **Step 5: Run** — `cd backend && uv run pytest tests/integration -q` (Docker Postgres up) → PASS, plus `uv run pytest tests/unit -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/alembic/versions/0004_rule_schedule.py backend/app/models/tables.py backend/app/schemas backend/app/api/routes_events.py backend/app/repositories/rules.py backend/app/alerts/event_service.py backend/tests/integration/test_api.py
git commit -m "feat(phase4): persist rule schedules; behaviour checks in API, presets and events"
```

---

### Task 3: Fall (temporal sequence) and stalling assessors

**Files:**
- Create: `backend/app/behaviour/__init__.py` (empty docstring module for now; factory in Task 6)
- Create: `backend/app/behaviour/assessors.py`
- Test: `backend/tests/unit/test_behaviour.py`

**Interfaces:**
- Consumes: `BehaviourConfig`, `BEHAVIOURS`, `RuleThresholds` (Task 1); `TrackedPerson`, `Observation`, `Outcome` (`app/ppe/types.py`); `PPEStateMachine` (unchanged).
- Produces (later tasks rely on these exact names):
  - `gate(p: TrackedPerson, t: RuleThresholds) -> str | None`
  - `class BehaviourAssessor` with `set_thresholds(t)`, `take_pending() -> list` (always `[]`), `assess_deferred(persons, dets, frame) -> list[Observation]`, hooks `_one(p, dets, frame) -> Observation` and `_drop(track_id)`.
  - `FallAssessor(cfg: BehaviourConfig, thresholds: RuleThresholds)` — reasons: `upright`, `transition`, `down_after_fall`, `horizontal_without_fall`, `not_assessable:torso_not_visible`.
  - `IdleAssessor(cfg, thresholds)` — reasons: `warming_up`, `still`, `moving`.

**Design notes for the implementer:**
- Outcome mapping for every behaviour: `ABSENT` = behaviour happening (violation evidence), `PRESENT` = normal, `UNCERTAIN` = not assessable. The existing `PPEStateMachine` then confirms with the rule's policy (fall: `duration`, 3 s held).
- Fall = **fast** sequence: an upright sample (torso ≤ `upright_max_deg` from vertical) within the last `fall_window_s`, then a lying sample (≥ `lying_min_deg`) with the hip centre ≥ `drop_frac` × standing height lower. Lying down slowly (upright sample already out of the window), bending (hips do not drop) and squatting (torso stays upright) never start an episode.
- Per-track state is pruned for tracks unseen for `FORGET_S` (bounded memory).

- [ ] **Step 1: Write the failing tests** — create `backend/tests/unit/test_behaviour.py`:

```python
"""Phase 4 behaviour assessors: fall (temporal sequence), stalling (stillness), phone, fight."""

import numpy as np

from app.behaviour.assessors import FallAssessor, IdleAssessor
from app.config.settings import BEHAVIOURS, BehaviourConfig, RuleThresholds
from app.ppe.state_machine import PPEStateMachine
from app.ppe.types import Outcome, PPEState, TrackedPerson
from app.streaming.frame import Frame

CFG = BehaviourConfig()
IMG = np.zeros((720, 1280, 3), np.uint8)
UPRIGHT_BOX, LYING_BOX = (450.0, 150.0, 550.0, 550.0), (350.0, 480.0, 650.0, 560.0)


def frame(t):
    return Frame("cam01", int(t * 10), IMG, None, t, t)


def torso_kps(sh, hip, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[5], k[6] = (sh[0] - 20, sh[1], conf), (sh[0] + 20, sh[1], conf)
    k[11], k[12] = (hip[0] - 15, hip[1], conf), (hip[0] + 15, hip[1], conf)
    return k


UPRIGHT = (UPRIGHT_BOX, torso_kps((500, 200), (500, 350)))
MID = (UPRIGHT_BOX, torso_kps((440, 380), (520, 450)))  # ~49 deg: between upright and lying
LYING = (LYING_BOX, torso_kps((380, 520), (530, 520)))  # 90 deg, hips 170 px lower (> 0.25 * 400)
BENT = (UPRIGHT_BOX, torso_kps((380, 345), (500, 350)))  # torso horizontal, hips did not drop


def run(assessor, poses, dt=0.2, fsm=None):
    out = []
    for i, (box, kps) in enumerate(poses):
        t = i * dt
        (o,) = assessor.assess_deferred([TrackedPerson(1, box, 0.9, kps)], [], frame(t))
        trs = fsm.update([o], t) if fsm else []
        out.append((t, o, trs))
    return out


def test_fast_fall_then_lying_is_absent_and_confirms_after_hold():
    fsm = PPEStateMachine("cam01", BEHAVIOURS["fall"].policy, 3.0)
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] * 3 + [MID] + [LYING] * 20, fsm=fsm)
    assert [o.reason for _, o, _ in res[:4]] == ["upright"] * 3 + ["transition"]
    assert all(o.outcome is Outcome.ABSENT and o.reason == "down_after_fall" for _, o, _ in res[4:])
    confirmed = [t for t, _, trs in res for tr in trs if tr.new is PPEState.CONFIRMED_NON_COMPLIANT]
    assert confirmed and 3.8 - 1e-6 <= confirmed[0] <= 4.0 + 1e-6  # first ABSENT at 0.8 s + 3 s hold


def test_getting_up_ends_the_fall():
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] * 3 + [LYING] * 3 + [UPRIGHT])
    assert res[-1][1].outcome is Outcome.PRESENT and res[-1][1].reason == "upright"


def test_lying_down_slowly_never_starts_an_episode():
    fsm = PPEStateMachine("cam01", BEHAVIOURS["fall"].policy, 3.0)
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] + [MID] * 15 + [LYING] * 30, fsm=fsm)
    assert all(o.outcome is not Outcome.ABSENT for _, o, _ in res)
    assert res[-1][1].reason == "horizontal_without_fall"
    assert not any(tr.new is PPEState.CONFIRMED_NON_COMPLIANT for _, _, trs in res for tr in trs)


def test_bending_over_is_not_a_fall():
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] * 3 + [BENT] * 5)
    assert all(o.outcome is Outcome.PRESENT for _, o, _ in res)


def test_no_pose_is_uncertain_never_a_fall():
    (o,) = FallAssessor(CFG, RuleThresholds()).assess_deferred([TrackedPerson(1, UPRIGHT_BOX, 0.9, None)], [],
                                                                frame(0.0))
    assert o.outcome is Outcome.UNCERTAIN and o.reason == "not_assessable:torso_not_visible"


def test_small_or_low_confidence_person_is_not_assessed():
    a = FallAssessor(CFG, RuleThresholds())
    (o,) = a.assess_deferred([TrackedPerson(1, (0, 0, 20, 30), 0.9, UPRIGHT[1])], [], frame(0.0))
    assert o.reason == "not_assessable:person_too_small"
    (o,) = a.assess_deferred([TrackedPerson(2, UPRIGHT_BOX, 0.1, UPRIGHT[1])], [], frame(0.0))
    assert o.reason == "not_assessable:low_person_conf"


def body_kps(jitter=0.0, wrist_dx=0.0, rng=None):
    k = np.zeros((17, 3), np.float32)
    pts = {0: (460, 130), 5: (420, 160), 6: (500, 160), 7: (410, 240), 8: (510, 240), 9: (405 + wrist_dx, 300),
           10: (515, 300), 11: (430, 300), 12: (490, 300), 15: (440, 480), 16: (480, 480)}
    for i, (x, y) in pts.items():
        j = rng.uniform(-jitter, jitter, 2) if rng is not None else (0, 0)
        k[i] = (x + j[0], y + j[1], 0.9)
    return k


def idle_run(kps_at, seconds, dt=0.2, fsm=None):
    a, out = IdleAssessor(CFG, RuleThresholds()), []
    for i in range(int(seconds / dt) + 1):
        t = i * dt
        (o,) = a.assess_deferred([TrackedPerson(1, (400.0, 100.0, 520.0, 500.0), 0.9, kps_at(t))], [], frame(t))
        out.append((t, o, fsm.update([o], t) if fsm else []))
    return out


def test_still_person_is_absent_after_warmup():
    rng = np.random.default_rng(0)
    res = idle_run(lambda t: body_kps(jitter=1.0, rng=rng), 15)
    assert res[0][1].reason == "warming_up" and res[-1][1].outcome is Outcome.ABSENT
    assert res[-1][1].reason == "still"


def test_working_hands_count_as_moving():
    res = idle_run(lambda t: body_kps(wrist_dx=40 * np.sin(t * 3)), 15)
    assert res[-1][1].outcome is Outcome.PRESENT and res[-1][1].reason == "moving"


def test_stalling_confirms_after_the_rules_idle_time():
    policy = BEHAVIOURS["stalling"].policy.model_copy(update={"min_duration_s": 20.0})
    fsm = PPEStateMachine("cam01", policy, 3.0)
    res = idle_run(lambda t: body_kps(), 40, fsm=fsm)
    confirmed = [t for t, _, trs in res for tr in trs if tr.new is PPEState.CONFIRMED_NON_COMPLIANT]
    assert confirmed and 28.0 <= confirmed[0] <= 30.0  # ~8 s warm-up + 20 s idle
```

- [ ] **Step 2: Run to verify failure** — `cd backend && uv run pytest tests/unit/test_behaviour.py -q` → FAIL (`ModuleNotFoundError: app.behaviour`).

- [ ] **Step 3: Implement** — create `backend/app/behaviour/__init__.py`:

```python
"""Phase 4 behaviour checks (ADR-015)."""
```

Create `backend/app/behaviour/assessors.py`:

```python
"""Phase 4 behaviour assessors (ADR-015) with the Associator interface, so the PPE state machine, events, dedupe,
schedules and alerts apply unchanged. Outcome mapping: ABSENT = the behaviour is happening (violation evidence),
PRESENT = normal, UNCERTAIN = not assessable (never a violation)."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from app.config.settings import BehaviourConfig, RuleThresholds
from app.inference.types import Detection
from app.ppe.types import Observation, Outcome, TrackedPerson
from app.streaming.frame import Frame

KP_LSH, KP_RSH, KP_LHIP, KP_RHIP = 5, 6, 11, 12
FORGET_S = 30.0  # per-track history of tracks unseen this long is dropped (bounded memory)


def gate(p: TrackedPerson, t: RuleThresholds) -> str | None:
    """Same person gates as the PPE rules: unreliable or tiny detections are never assessed."""
    if p.conf < t.person_conf:
        return "low_person_conf"
    if p.box[3] - p.box[1] < t.min_person_height_px:
        return "person_too_small"
    return None


def _obs(p: TrackedPerson, outcome: Outcome, conf: float, reason: str) -> Observation:
    return Observation(track_id=p.track_id, outcome=outcome, confidence=conf, reason=reason, person_box=p.box,
                       head_roi=None)


class BehaviourAssessor:
    """Shared plumbing: thresholds, person gate, per-track pruning. Subclasses implement _one() (and _drop())."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds):
        self.cfg, self.t = cfg, thresholds
        self.seen: dict[int, float] = {}

    def set_thresholds(self, thresholds: RuleThresholds) -> None:
        self.t = thresholds

    def take_pending(self) -> list:
        return []  # no classifier stage

    def _drop(self, track_id: int) -> None:
        pass

    def _forget(self, now: float) -> None:
        for tid in [k for k, last in self.seen.items() if now - last > FORGET_S]:
            del self.seen[tid]
            self._drop(tid)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        self._forget(frame.captured_at)
        out = []
        for p in persons:
            self.seen[p.track_id] = frame.captured_at
            reason = gate(p, self.t)
            out.append(_obs(p, Outcome.UNCERTAIN, 0.0, f"not_assessable:{reason}") if reason
                       else self._one(p, dets, frame))
        return out

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        raise NotImplementedError


def torso(p: TrackedPerson, min_conf: float) -> tuple[float, float, float] | None:
    """(torso angle from vertical in degrees, hip-centre y, mean keypoint conf); None without shoulders + hips."""
    k = p.keypoints
    if k is None or min(k[i, 2] for i in (KP_LSH, KP_RSH, KP_LHIP, KP_RHIP)) < min_conf:
        return None
    sx, sy = (k[KP_LSH, 0] + k[KP_RSH, 0]) / 2, (k[KP_LSH, 1] + k[KP_RSH, 1]) / 2
    hx, hy = (k[KP_LHIP, 0] + k[KP_RHIP, 0]) / 2, (k[KP_LHIP, 1] + k[KP_RHIP, 1]) / 2
    if math.hypot(sx - hx, sy - hy) < 1e-6:
        return None
    conf = float(np.mean([k[i, 2] for i in (KP_LSH, KP_RSH, KP_LHIP, KP_RHIP)]))
    return math.degrees(math.atan2(abs(sx - hx), abs(sy - hy))), float(hy), conf


class FallAssessor(BehaviourAssessor):
    """Fall = a FAST transition: torso upright -> lying within fall_window_s while the hips drop >= drop_frac of the
    standing body height. Lying down slowly, bending (hips stay up) and squatting (torso stays upright) never start an
    episode. After a fall, ABSENT while the person stays lying; the rule's duration policy confirms 'still down'."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds):
        super().__init__(cfg, thresholds)
        self.hist: dict[int, deque[tuple[float, float, float]]] = {}  # (ts, angle, hip_y) within fall_window_s
        self.stand_h: dict[int, float] = {}  # person box height at the last upright sample
        self.fallen: set[int] = set()

    def _drop(self, track_id: int) -> None:
        self.hist.pop(track_id, None)
        self.stand_h.pop(track_id, None)
        self.fallen.discard(track_id)

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        cfg, now, tid = self.cfg, frame.captured_at, p.track_id
        tr = torso(p, cfg.keypoint_min_conf)
        if tr is None:
            return _obs(p, Outcome.UNCERTAIN, 0.0, "not_assessable:torso_not_visible")
        angle, hip_y, conf = tr
        h = self.hist.setdefault(tid, deque())
        h.append((now, angle, hip_y))
        while now - h[0][0] > cfg.fall_window_s:
            h.popleft()
        if angle <= cfg.upright_max_deg:
            self.stand_h[tid] = p.box[3] - p.box[1]
            self.fallen.discard(tid)
            return _obs(p, Outcome.PRESENT, conf, "upright")
        if angle < cfg.lying_min_deg:
            return _obs(p, Outcome.UNCERTAIN, 0.0, "transition")  # wait for the next sample
        stand = self.stand_h.get(tid)
        if tid not in self.fallen and stand and any(
                a <= cfg.upright_max_deg and hip_y - y >= cfg.drop_frac * stand for _, a, y in h):
            self.fallen.add(tid)
        if tid in self.fallen:
            return _obs(p, Outcome.ABSENT, conf, "down_after_fall")
        return _obs(p, Outcome.PRESENT, conf, "horizontal_without_fall")


class IdleAssessor(BehaviourAssessor):
    """Stalling = the body barely moves: over idle_window_s every keypoint (and the box centre) stays within
    idle_max_range body heights (5th-95th percentile range, robust to single-frame keypoint glitches). Hands-only
    work counts as moving. Measures stillness, not intent: scope the rule with a work-area zone and a schedule."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds):
        super().__init__(cfg, thresholds)
        self.hist: dict[int, deque[tuple[float, np.ndarray, float]]] = {}  # (ts, (18,2) pixels/NaN, box height)

    def _drop(self, track_id: int) -> None:
        self.hist.pop(track_id, None)

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        cfg, now = self.cfg, frame.captured_at
        pts = np.full((18, 2), np.nan, np.float32)
        pts[0] = ((p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2)
        k = p.keypoints
        if k is not None:
            ok = k[:, 2] >= cfg.keypoint_min_conf
            pts[1:][ok] = k[ok, :2]
        h = self.hist.setdefault(p.track_id, deque())
        h.append((now, pts, max(p.box[3] - p.box[1], 1e-6)))
        while now - h[0][0] > cfg.idle_window_s:
            h.popleft()
        if now - h[0][0] < 0.8 * cfg.idle_window_s:
            return _obs(p, Outcome.UNCERTAIN, 0.0, "warming_up")
        arr = np.stack([x for _, x, _ in h])  # (T, 18, 2)
        valid = (~np.isnan(arr[..., 0])).sum(axis=0) >= 2
        lo = np.nanpercentile(arr[:, valid], 5, axis=0)
        hi = np.nanpercentile(arr[:, valid], 95, axis=0)
        span = (hi - lo).T  # (2, V)
        move = float(np.max(np.hypot(span[0], span[1]))) / float(np.median([b for _, _, b in h]))
        if move <= cfg.idle_max_range:
            return _obs(p, Outcome.ABSENT, 1.0 - move / cfg.idle_max_range, "still")
        return _obs(p, Outcome.PRESENT, min(1.0, move / cfg.idle_max_range - 1.0), "moving")
```

- [ ] **Step 4: Run** — `cd backend && uv run pytest tests/unit/test_behaviour.py -q` → PASS. If `test_stalling_confirms_after_the_rules_idle_time` lands just outside `[28, 30]`, print the transitions and fix the code, not the window, unless the arithmetic (warm-up = 0.8 × 10 s, then `suspect_n=3` samples, then 20 s held) proves the window wrong.

- [ ] **Step 5: Commit**

```bash
git add backend/app/behaviour backend/tests/unit/test_behaviour.py
git commit -m "feat(phase4): fall (temporal sequence) and stalling assessors"
```

---

### Task 4: Phone stage and phone assessor

**Files:**
- Modify: `backend/app/inference/ultralytics_detector.py`
- Modify: `backend/app/behaviour/assessors.py` (add `PhoneAssessor`)
- Test: `backend/tests/unit/test_behaviour.py` (append)

**Interfaces:**
- Consumes: `ModelsConfig.phone`, `phone_class_map` (Task 1); `region_rois`, `head_roi` (`app/ppe/head_roi.py`); `ioa`, `clip` (`app/ppe/geometry.py`); `AssociationConfig`.
- Produces: detections with `cls == "phone"` from the cascade when `models.phone` is set; `PhoneAssessor(cfg: BehaviourConfig, thresholds: RuleThresholds, assoc: AssociationConfig)` — reasons `phone_at_hand_or_head`, `no_phone`; ABSENT observations carry `item_roi` = `absent_box` = the phone box (drawn on evidence).

- [ ] **Step 1: Write the failing tests** — append to `backend/tests/unit/test_behaviour.py`:

```python
from app.behaviour.assessors import PhoneAssessor  # noqa: E402
from app.config.settings import AssociationConfig  # noqa: E402
from app.inference.types import Detection  # noqa: E402

PERSON = (400.0, 100.0, 520.0, 500.0)
IN_HAND = (390.0, 300.0, 420.0, 340.0)  # over the left hand ROI (~377,295,429,347)


def phone_at(a, t, phone=None, conf=0.9):
    dets = [Detection("phone", conf, phone, "t")] if phone else []
    (o,) = a.assess_deferred([TrackedPerson(1, PERSON, 0.9, body_kps())], dets, frame(t))
    return o


def test_phone_in_hand_is_absent_with_the_phone_as_evidence():
    o = phone_at(PhoneAssessor(CFG, RuleThresholds(), AssociationConfig()), 0.0, IN_HAND)
    assert o.outcome is Outcome.ABSENT and o.reason == "phone_at_hand_or_head" and o.item_roi == IN_HAND


def test_phone_far_from_the_body_or_low_confidence_is_normal():
    a = PhoneAssessor(CFG, RuleThresholds(), AssociationConfig())
    assert phone_at(a, 0.0, (800.0, 600.0, 830.0, 640.0)).outcome is Outcome.PRESENT
    assert phone_at(a, 0.2, IN_HAND, conf=0.2).outcome is Outcome.PRESENT  # below the rule's absent_conf


def test_short_detector_dropouts_are_bridged():
    a = PhoneAssessor(CFG, RuleThresholds(), AssociationConfig())
    assert phone_at(a, 0.0, IN_HAND).outcome is Outcome.ABSENT
    assert phone_at(a, 1.0).outcome is Outcome.ABSENT  # 1 s gap <= phone_gap_s (2 s)
    assert phone_at(a, 2.5).outcome is Outcome.PRESENT
```

- [ ] **Step 2: Run to verify failure** — `cd backend && uv run pytest tests/unit/test_behaviour.py -k phone -q` → FAIL (`ImportError: PhoneAssessor`).

- [ ] **Step 3: Implement `PhoneAssessor`** — in `assessors.py` add imports
`from app.config.settings import AssociationConfig, BehaviourConfig, RuleThresholds`,
`from app.ppe.geometry import clip, ioa`, `from app.ppe.head_roi import head_roi, region_rois`,
`from app.ppe.types import Box, Observation, Outcome, TrackedPerson`, then append:

```python
def _grow(b: Box, s: float, w: float, h: float) -> Box:
    cx, cy, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * s / 2, (b[3] - b[1]) * s / 2
    return clip((cx - hw, cy - hh, cx + hw, cy + hh), w, h)


class PhoneAssessor(BehaviourAssessor):
    """Phone use = a detected phone over a hand or at the head (call). Dropouts shorter than phone_gap_s keep the
    evidence, so a small, intermittently detected phone still builds an episode. Evidence conf = rule absent_conf."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds, assoc: AssociationConfig):
        super().__init__(cfg, thresholds)
        self.assoc = assoc
        self.last: dict[int, tuple[float, Box, float]] = {}  # track -> (ts, phone box, conf) of the last match

    def _drop(self, track_id: int) -> None:
        self.last.pop(track_id, None)

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        cfg, now, w, h = self.cfg, frame.captured_at, frame.width, frame.height
        rois = [r for r, _ in region_rois("hands", p, self.assoc, w, h)] + [head_roi(p, self.assoc, w, h)[0]]
        grown = [_grow(r, cfg.phone_roi_scale, w, h) for r in rois]
        phones = [d for d in dets if d.cls == "phone" and d.conf >= self.t.absent_conf]
        best = max(((ioa(d.xyxy, r), d) for d in phones for r in grown), key=lambda x: x[0], default=(0.0, None))
        if best[1] is not None and best[0] >= cfg.phone_min_ioa:
            self.last[p.track_id] = (now, best[1].xyxy, best[1].conf)
        seen = self.last.get(p.track_id)
        if seen and now - seen[0] <= cfg.phone_gap_s:
            o = _obs(p, Outcome.ABSENT, seen[2], "phone_at_hand_or_head")
            o.item_roi = o.absent_box = seen[1]
            return o
        return _obs(p, Outcome.PRESENT, 1.0, "no_phone")
```

- [ ] **Step 4: Detector stage** — in `UltralyticsCascadeDetector.__init__` after the items lines:

```python
        # Phase 4: COCO detector for phones (ADR-015); same downscaled frames, one more batched call
        self.phone = _Stage(models.phone, models_dir) if models.phone else None
        self._phone_ids = {v: k for k, v in models.phone_class_map.items()}
        if self.phone:
            self.version += f"+{self.phone.version}"
```

`model_versions` return: add `**({"phone": self.phone.version} if self.phone else {})`.
`warmup` loop and `_fit` stage tuples: `(self.person, self.ppe, self.items, self.phone)`.
In `detect`, the stage loop tuple becomes:
`(("ppe", self.ppe, self._ppe_ids), ("items", self.items, self._items_ids), ("phone", self.phone, self._phone_ids))`.
In `pipeline_worker.py` add `"phone"` to the `stage_ms` key tuple (`("person", "ppe", "items", "phone", ...)`) so its timing is reported.

- [ ] **Step 5: Run** — `cd backend && uv run pytest tests/unit -q` → PASS. With a GPU: `uv run pytest tests -q -m gpu` → existing cascade test still PASS (phone unset by default).

- [ ] **Step 6: Commit**

```bash
git add backend/app/behaviour/assessors.py backend/app/inference/ultralytics_detector.py backend/app/workers/pipeline_worker.py backend/tests/unit/test_behaviour.py
git commit -m "feat(phase4): phone detector stage and phone-use assessor"
```

---

### Task 5: Fight model and fight assessor

**Files:**
- Create: `backend/app/behaviour/fight.py`
- Test: `backend/tests/unit/test_fight.py`

**Interfaces:**
- Consumes: `BehaviourAssessor`, `gate`, `_obs` (Task 3); `BehaviourConfig.fight_pair_dist`; `weights_version` (`app/inference/ultralytics_detector.py`).
- Produces:
  - `SEQ_LEN = 16`, `HZ = 5.0` (training defaults; the checkpoint stores the values actually used)
  - `pair_features(ka, kb, box_a, box_b, min_conf=0.3) -> np.ndarray` shape `(102,)`, order-invariant
  - `PairSequencer(seq_len, hz, pair_dist)` with `.update(persons, now) -> tuple[set[int], list[tuple[tuple[int, int], np.ndarray]]]` (`np.ndarray` shape `(seq_len, 102)`) and `.forget(track_id)`
  - `FightGRU(hidden)`; `FightModel(path)` with `.seq_len`, `.hz`, `.version`, `.score(x: (B, T, 102) float32) -> (B,) probs`
  - `FightAssessor(cfg, thresholds, model: FightModel | None)` — reasons `fight_model_missing`, `fight_score`, `pair_normal`, `warming_up`, `no_one_nearby`
- Training scripts (Task 9) import `pair_features`, `PairSequencer`, `FightGRU`, `SEQ_LEN`, `HZ` from here, so train and serve cannot drift.

- [ ] **Step 1: Write the failing tests** — create `backend/tests/unit/test_fight.py`:

```python
"""Phase 4 fight check: order-invariant pair features, fixed-rate pair sequences, model round trip, assessor."""

import numpy as np
import torch

from app.behaviour.fight import FightAssessor, FightGRU, FightModel, PairSequencer, pair_features
from app.config.settings import BehaviourConfig, RuleThresholds
from app.ppe.types import Outcome, TrackedPerson
from app.streaming.frame import Frame

IMG = np.zeros((720, 1280, 3), np.uint8)
A, B, FAR = (400.0, 100.0, 520.0, 500.0), (500.0, 100.0, 620.0, 500.0), (1000.0, 100.0, 1120.0, 500.0)


def kps(dx=0.0, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[:, 0], k[:, 1], k[:, 2] = np.linspace(410, 510, 17) + dx, np.linspace(120, 480, 17), conf
    return k


class Stub:
    def __init__(self, prob, seq_len=4, hz=5.0):
        self.prob, self.seq_len, self.hz, self.version = prob, seq_len, hz, "stub"

    def score(self, x):
        assert x.shape[1:] == (self.seq_len, 102)
        return np.full(len(x), self.prob, np.float32)


def run(assessor, people, seconds=1.2, dt=0.2):
    out = []
    for i in range(int(seconds / dt) + 1):
        t = i * dt
        persons = [TrackedPerson(tid, box, 0.9, kps(box[0] - 400)) for tid, box in people]
        out = assessor.assess_deferred(persons, [], Frame("cam01", i, IMG, None, t, t))
    return out


def test_pair_features_are_order_invariant_and_drop_low_confidence_points():
    f1 = pair_features(kps(), kps(100), A, B)
    assert f1.shape == (102,) and np.allclose(f1, pair_features(kps(100), kps(), B, A))
    low = pair_features(kps(conf=0.1), kps(100), A, B)
    assert np.all(low[:51] == 0)  # left person's points below min_conf are zeroed


def test_sequencer_needs_a_full_window_and_a_nearby_partner():
    sq = PairSequencer(seq_len=4, hz=5.0, pair_dist=1.2)
    for i in range(3):
        near, seqs = sq.update([TrackedPerson(1, A, 0.9, kps()), TrackedPerson(2, B, 0.9, kps(100))], i * 0.2)
        assert near == {1, 2} and seqs == []  # 3 samples < seq_len
    near, seqs = sq.update([TrackedPerson(1, A, 0.9, kps()), TrackedPerson(2, B, 0.9, kps(100))], 0.6)
    assert [p for p, _ in seqs] == [(1, 2)] and seqs[0][1].shape == (4, 102)
    near, seqs = sq.update([TrackedPerson(1, A, 0.9, kps()), TrackedPerson(3, FAR, 0.9, kps(600))], 0.8)
    assert near == set() and seqs == []


def test_assessor_outcomes():
    cfg, t = BehaviourConfig(), RuleThresholds(absent_conf=0.7)
    hot = run(FightAssessor(cfg, t, Stub(0.9)), [(1, A), (2, B)])
    assert [o.outcome for o in hot] == [Outcome.ABSENT] * 2 and hot[0].reason == "fight_score"
    calm = run(FightAssessor(cfg, t, Stub(0.1)), [(1, A), (2, B)])
    assert {o.reason for o in calm} == {"pair_normal"}
    alone = run(FightAssessor(cfg, t, Stub(0.9)), [(1, A), (3, FAR)])
    assert {o.outcome for o in alone} == {Outcome.PRESENT} and {o.reason for o in alone} == {"no_one_nearby"}
    early = run(FightAssessor(cfg, t, Stub(0.9)), [(1, A), (2, B)], seconds=0.2)
    assert {o.reason for o in early} == {"warming_up"}
    none = run(FightAssessor(cfg, t, None), [(1, A), (2, B)])
    assert {o.reason for o in none} == {"fight_model_missing"} and none[0].outcome is Outcome.UNCERTAIN


def test_model_checkpoint_round_trip(tmp_path):
    p = tmp_path / "fight.pt"
    torch.save({"state_dict": FightGRU(8).state_dict(), "hidden": 8, "seq_len": 4, "hz": 5.0}, p)
    m = FightModel(p)
    probs = m.score(np.zeros((2, 4, 102), np.float32))
    assert m.seq_len == 4 and m.hz == 5.0 and probs.shape == (2,) and np.all((probs > 0) & (probs < 1))
```

- [ ] **Step 2: Run to verify failure** — `cd backend && uv run pytest tests/unit/test_fight.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** — create `backend/app/behaviour/fight.py`:

```python
"""Fighting (Phase 4, ADR-015): a small GRU scores the keypoint sequences of two nearby tracked people.

pair_features / PairSequencer are used by BOTH this assessor and scripts/ml/extract_fight_sequences.py, so training
and production features cannot drift. Sequences are resampled to the checkpoint's rate (hz), so the model does not
depend on the camera's processed FPS - which must be >= hz (else pairs stay 'warming_up' = UNCERTAIN)."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.behaviour.assessors import BehaviourAssessor, _obs, gate
from app.config.settings import BehaviourConfig, RuleThresholds
from app.inference.types import Detection
from app.inference.ultralytics_detector import weights_version
from app.ppe.types import Box, Observation, Outcome, TrackedPerson
from app.streaming.frame import Frame

SEQ_LEN, HZ = 16, 5.0  # training defaults (3 s of motion at 5 Hz); checkpoints store the values used
N_FEAT = 102  # 2 people x 17 keypoints x (x, y, conf)


def pair_features(ka: np.ndarray | None, kb: np.ndarray | None, box_a: Box, box_b: Box,
                  min_conf: float = 0.3) -> np.ndarray:
    """Keypoints of two people -> (102,): x, y relative to the pair centre in mean body heights, plus confidence.
    Left person first (order-invariant); points below min_conf are zeroed."""
    if box_a[0] + box_a[2] > box_b[0] + box_b[2]:
        ka, kb, box_a, box_b = kb, ka, box_b, box_a
    h = max(((box_a[3] - box_a[1]) + (box_b[3] - box_b[1])) / 2, 1e-6)
    cx = (box_a[0] + box_a[2] + box_b[0] + box_b[2]) / 4
    cy = (box_a[1] + box_a[3] + box_b[1] + box_b[3]) / 4
    out = np.zeros((34, 3), np.float32)
    for i, k in enumerate((ka, kb)):
        if k is None:
            continue
        ok = k[:, 2] >= min_conf
        seg = out[17 * i:17 * i + 17]
        seg[ok, 0] = (k[ok, 0] - cx) / h
        seg[ok, 1] = (k[ok, 1] - cy) / h
        seg[ok, 2] = k[ok, 2]
    return out.reshape(-1)


def _nearest(h: deque, t: float, tol: float):
    best = min(h, key=lambda s: abs(s[0] - t), default=None)
    return best if best is not None and abs(best[0] - t) <= tol else None


class PairSequencer:
    """Per-camera keypoint history -> fixed-rate pair sequences for people closer than pair_dist body heights."""

    def __init__(self, seq_len: int, hz: float, pair_dist: float):
        self.seq_len, self.hz, self.pair_dist = seq_len, hz, pair_dist
        self.hist: dict[int, deque[tuple[float, Box, np.ndarray | None]]] = {}

    @property
    def span(self) -> float:
        return (self.seq_len - 1) / self.hz

    def forget(self, track_id: int) -> None:
        self.hist.pop(track_id, None)

    def _sequence(self, ta: int, tb: int, now: float) -> np.ndarray | None:
        tol, out = 0.6 / self.hz, []
        for i in range(self.seq_len):
            t = now - (self.seq_len - 1 - i) / self.hz
            sa, sb = _nearest(self.hist[ta], t, tol), _nearest(self.hist[tb], t, tol)
            if sa is None or sb is None:
                return None
            out.append(pair_features(sa[2], sb[2], sa[1], sb[1]))
        return np.stack(out)

    def update(self, persons: list[TrackedPerson],
               now: float) -> tuple[set[int], list[tuple[tuple[int, int], np.ndarray]]]:
        """Record persons; return (track ids with a nearby person, scorable (pair, sequence) items)."""
        for p in persons:
            h = self.hist.setdefault(p.track_id, deque())
            h.append((now, p.box, p.keypoints))
            while now - h[0][0] > self.span + 1.0:
                h.popleft()
        near: set[int] = set()
        out = []
        for i, a in enumerate(persons):
            for b in persons[i + 1:]:
                mean_h = max(((a.box[3] - a.box[1]) + (b.box[3] - b.box[1])) / 2, 1e-6)
                d = math.hypot((a.box[0] + a.box[2] - b.box[0] - b.box[2]) / 2,
                               (a.box[1] + a.box[3] - b.box[1] - b.box[3]) / 2)
                if d > self.pair_dist * mean_h:
                    continue
                near |= {a.track_id, b.track_id}
                seq = self._sequence(a.track_id, b.track_id, now)
                if seq is not None:
                    out.append(((a.track_id, b.track_id), seq))
        return near, out


class FightGRU(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 102) -> (B,) logits
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class FightModel:
    """Checkpoint on CPU: a GRU over 16 x 102 floats costs microseconds; no GPU sync on the inference thread."""

    def __init__(self, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.seq_len, self.hz = int(ck["seq_len"]), float(ck["hz"])
        self.net = FightGRU(int(ck["hidden"]))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.version = weights_version(Path(path))

    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        return torch.sigmoid(self.net(torch.from_numpy(np.ascontiguousarray(x, np.float32)))).numpy()


class FightAssessor(BehaviourAssessor):
    """Per track: max fight probability over its nearby pairs >= the rule's absent_conf -> ABSENT. Alone -> PRESENT
    (one person cannot fight). Partner present but sequence not yet full -> UNCERTAIN. No model -> UNCERTAIN."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds, model: FightModel | None):
        super().__init__(cfg, thresholds)
        self.model = model
        self.seq = PairSequencer(model.seq_len, model.hz, cfg.fight_pair_dist) if model else None

    def _drop(self, track_id: int) -> None:
        if self.seq:
            self.seq.forget(track_id)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        now = frame.captured_at
        self._forget(now)
        for p in persons:
            self.seen[p.track_id] = now
        if self.model is None or self.seq is None:
            return [_obs(p, Outcome.UNCERTAIN, 0.0, "fight_model_missing") for p in persons]
        ok = [p for p in persons if gate(p, self.t) is None]
        near, items = self.seq.update(ok, now)
        probs = self.model.score(np.stack([s for _, s in items])) if items else []
        best: dict[int, float] = {}
        for ((ta, tb), _), pr in zip(items, probs, strict=True):
            best[ta], best[tb] = max(best.get(ta, 0.0), float(pr)), max(best.get(tb, 0.0), float(pr))
        out = []
        for p in persons:
            reason = gate(p, self.t)
            if reason:
                out.append(_obs(p, Outcome.UNCERTAIN, 0.0, f"not_assessable:{reason}"))
            elif p.track_id in best:
                pr = best[p.track_id]
                out.append(_obs(p, Outcome.ABSENT, pr, "fight_score") if pr >= self.t.absent_conf
                           else _obs(p, Outcome.PRESENT, 1.0 - pr, "pair_normal"))
            elif p.track_id in near:
                out.append(_obs(p, Outcome.UNCERTAIN, 0.0, "warming_up"))
            else:
                out.append(_obs(p, Outcome.PRESENT, 1.0, "no_one_nearby"))
        return out
```

- [ ] **Step 4: Run** — `cd backend && uv run pytest tests/unit/test_fight.py tests/unit/test_behaviour.py -q` → PASS; `uv run ruff check app tests`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/behaviour/fight.py backend/tests/unit/test_fight.py
git commit -m "feat(phase4): fight model (pair keypoint GRU) and fight assessor"
```

---

### Task 6: Wire behaviour rules and schedules into the pipeline worker

**Files:**
- Modify: `backend/app/behaviour/__init__.py` (factory)
- Modify: `backend/app/workers/pipeline_worker.py`
- Test: `backend/tests/unit/test_behaviour_worker.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `make_behaviour_assessor(check, cfg, thresholds, assoc, fight_model) -> BehaviourAssessor`; `RuleRuntime.schedule: Schedule | None`; `PipelineWorker.fight: FightModel | None`; `PipelineWorker._assessor(r: RuleSeed)`.

- [ ] **Step 1: Write the failing tests** — create `backend/tests/unit/test_behaviour_worker.py`:

```python
"""Phase 4 worker wiring: behaviour rules get assessors; schedules gate rules by site-local time."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from app.behaviour.assessors import FallAssessor, PhoneAssessor
from app.camera.source import CameraRuntimeConfig
from app.config.settings import AppConfig, RuleSeed, Schedule
from app.inference.types import Detection
from app.observability.stats import RollingWindow
from app.ppe.association import Associator
from app.ppe.types import Outcome, TrackedPerson
from app.streaming.frame import Frame
from app.workers.pipeline_worker import CamRuntime, PipelineWorker

DXB = ZoneInfo("Asia/Dubai")


def body_kps():
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in {0: (460, 130), 5: (420, 160), 6: (500, 160), 7: (410, 240), 8: (510, 240), 9: (405, 300),
                      10: (515, 300), 11: (430, 300), 12: (490, 300), 15: (440, 480), 16: (480, 480)}.items():
        k[i] = (x, y, 0.9)
    return k


class PassThroughTracker:
    active_count = 0

    def update(self, persons, frame):
        return [TrackedPerson(i + 1, d.xyxy, d.conf, d.keypoints) for i, d in enumerate(persons)]


def bare_worker():
    w = object.__new__(PipelineWorker)  # no DB / GPU: only the rule-building and assessment paths are exercised
    w.cfg, w.classifier, w.fight = AppConfig(), None, None
    w.stage_ms = {k: RollingWindow(60.0) for k in ("tracking", "association")}
    return w


def cam(w, seeds):
    cr = CamRuntime(CameraRuntimeConfig("cam01", "rtsp://127.0.0.1:8554/cam01", None, "tcp", True, 5.0), "Cam 1")
    w._build_rules(cr, seeds)
    cr.tracker = PassThroughTracker()
    return cr


def test_rules_get_the_right_assessor():
    w = bare_worker()
    cr = cam(w, [RuleSeed(id="hat", name="Hat"), RuleSeed(id="fall", name="Fall", ppe_class="fall"),
                 RuleSeed(id="phone", name="Phone", ppe_class="phone")])
    kinds = {r.id: type(r.associator) for r in cr.rules}
    assert kinds == {"hat": Associator, "fall": FallAssessor, "phone": PhoneAssessor}


def test_schedule_gates_a_rule_by_local_time():
    w = bare_worker()
    sched = Schedule(tz="Asia/Dubai", days=[0], windows=["07:00-08:00"])
    cr = cam(w, [RuleSeed(id="phone", name="Phone", ppe_class="phone", schedule=sched)])
    dets = [Detection("person", 0.9, (400.0, 100.0, 520.0, 500.0), "t", body_kps()),
            Detection("phone", 0.9, (390.0, 300.0, 420.0, 340.0), "t")]
    img = np.zeros((720, 1280, 3), np.uint8)

    def obs_at(h, m):
        t = datetime(2026, 11, 23, h, m, tzinfo=DXB).timestamp()  # a Monday
        return w._track_assess(cr, Frame("cam01", 1, img, None, t, t), dets)["per_rule"][0][1]

    assert obs_at(9, 0) == []  # off shift: no observations, so open episodes expire and close
    (o,) = obs_at(7, 30)
    assert o.outcome is Outcome.ABSENT


def test_hot_reload_updates_the_schedule_in_place():
    w = bare_worker()
    cr = cam(w, [RuleSeed(id="fall", name="Fall", ppe_class="fall")])
    before = cr.rules[0]
    sched = Schedule(tz="Asia/Dubai", days=[0], windows=["07:00-08:00"])
    w._build_rules(cr, [RuleSeed(id="fall", name="Fall", ppe_class="fall", schedule=sched)], cr.rules)
    assert cr.rules[0] is before and before.schedule == sched  # track state kept, schedule applied
```

- [ ] **Step 2: Run to verify failure** — `cd backend && uv run pytest tests/unit/test_behaviour_worker.py -q` → FAIL (`AttributeError ... 'fight'` / wrong assessor types).

- [ ] **Step 3: Factory** — replace `backend/app/behaviour/__init__.py` with:

```python
"""Phase 4 behaviour checks (ADR-015): assessors with the Associator interface, chosen per rule."""

from __future__ import annotations

from app.behaviour.assessors import BehaviourAssessor, FallAssessor, IdleAssessor, PhoneAssessor
from app.behaviour.fight import FightAssessor, FightModel
from app.config.settings import AssociationConfig, BehaviourConfig, RuleThresholds


def make_behaviour_assessor(check: str, cfg: BehaviourConfig, thresholds: RuleThresholds, assoc: AssociationConfig,
                            fight_model: FightModel | None) -> BehaviourAssessor:
    if check == "fall":
        return FallAssessor(cfg, thresholds)
    if check == "stalling":
        return IdleAssessor(cfg, thresholds)
    if check == "phone":
        return PhoneAssessor(cfg, thresholds, assoc)
    if check == "fight":
        return FightAssessor(cfg, thresholds, fight_model)
    raise ValueError(f"unknown behaviour {check!r}")
```

- [ ] **Step 4: Worker changes** in `backend/app/workers/pipeline_worker.py`:

4a. Imports: add `from app.behaviour import make_behaviour_assessor`, `from app.behaviour.fight import FightModel`, and add `BEHAVIOURS`, `PPE_ITEMS`, `Schedule` to the `app.config.settings` import list.

4b. `RuleRuntime`: change `associator: Associator` to `associator: Any  # Associator (PPE) or BehaviourAssessor (Phase 4)` and add the last field `schedule: Schedule | None = None  # None = always enforced`.

4c. `__init__`, after `self.model_versions[...]` for the classifier:

```python
        # Phase 4 fight model (CPU, tiny). Configured but missing weights fail loudly at start, like detector weights.
        self.fight = FightModel(settings.resolve(settings.ppe_models_dir) / models_cfg.fight_weights) \
            if models_cfg.fight_weights else None
        if self.fight is not None:
            self.model_versions["fight"] = self.fight.version
```

4d. Add the method and use it in `_build_rules`:

```python
    def _assessor(self, r: RuleSeed):
        if r.ppe_class in BEHAVIOURS:
            return make_behaviour_assessor(r.ppe_class, self.cfg.behaviour, r.thresholds, self.cfg.association,
                                           self.fight)
        return Associator(self.cfg.association, r.thresholds, self.classifier, r.ppe_class)
```

In `_build_rules`, the hot-reload branch becomes
`rr.severity, rr.zone_id, rr.mandatory, rr.schedule = r.severity, r.zone_id, r.mandatory, r.schedule`
and the new-rule branch becomes:

```python
                rr = RuleRuntime(r.id, r.severity, r.zone_id, self._assessor(r),
                                 PPEStateMachine(cr.cfg.id, r.temporal_policy, self.cfg.tracking.track_expiry_s),
                                 r.ppe_class, r.mandatory, r.schedule)
```

4e. `_track_assess`, first lines inside `for rr in cr.rules:`:

```python
            if rr.schedule is not None and not rr.schedule.active(frame.captured_at):
                per_rule.append((rr, []))  # off shift: no observations; tracks expire, open episodes close
                continue
```

4f. `_decide`, teacher compliant sampling condition becomes
`elif tr.new is S.COMPLIANT and rr.ppe_class in PPE_ITEMS and random.random() < self.teacher_sample_rate:`.

- [ ] **Step 5: Run** — `cd backend && uv run pytest tests -q -m "not gpu"` (Docker Postgres up) → all PASS; `uv run ruff check app tests`.

- [ ] **Step 6: Manual smoke (demo stack)** — start the demo (`python scripts/start_demo.py`), in Rules & settings create a `fall` rule on cam01 and a `phone` rule with a schedule; confirm the worker log shows `rules reloaded` and no exception; live overlay still renders. (Mock-detector demo frames have no keypoints, so behaviour rules stay UNCERTAIN there — expected.)

- [ ] **Step 7: Commit**

```bash
git add backend/app/behaviour/__init__.py backend/app/workers/pipeline_worker.py backend/tests/unit/test_behaviour_worker.py
git commit -m "feat(phase4): behaviour rules and shift schedules in the pipeline worker"
```

---

### Task 7: UI — behaviour checks, shift schedule editor, labels

**Files:**
- Modify: `frontend/src/types/api.ts`, `frontend/src/services/api.ts`, `frontend/src/features/Admin.tsx`, `frontend/src/components/LiveView.tsx`, `frontend/src/features/Events.tsx`
- Test: `frontend/src/app.test.tsx` (append)

**Interfaces:**
- Consumes: `GET /api/v1/ppe-items` → `{id, label, region | null, kind}`; rule `schedule` (Task 2).
- Produces: exported `ScheduleEditor({ value, onChange })` and `DEFAULT_SCHEDULE` from `features/Admin.tsx`; `Schedule` type and `BEHAVIOUR_IDS` from `types/api.ts`.

- [ ] **Step 1: Write the failing test** — append to `frontend/src/app.test.tsx` (and add `ScheduleEditor, DEFAULT_SCHEDULE` to the existing `./features/Admin` import):

```tsx
describe("ScheduleEditor", () => {
  it("enables a default shift, toggles days and parses windows on blur", () => {
    const onChange = vi.fn();
    const { rerender } = render(<ScheduleEditor value={null} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/only during shift hours/));
    expect(onChange).toHaveBeenLastCalledWith(DEFAULT_SCHEDULE);
    rerender(<ScheduleEditor value={DEFAULT_SCHEDULE} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText("Mon"));
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_SCHEDULE, days: DEFAULT_SCHEDULE.days.filter((d) => d !== 0) });
    const windows = screen.getByLabelText(/Shift windows/);
    fireEvent.change(windows, { target: { value: "06:00-14:00  22:00-06:00 " } });
    fireEvent.blur(windows);
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_SCHEDULE, windows: ["06:00-14:00", "22:00-06:00"] });
    fireEvent.click(screen.getByLabelText(/only during shift hours/));
    expect(onChange).toHaveBeenLastCalledWith(null);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `cd frontend && npx vitest run` → FAIL (`ScheduleEditor` is not exported).

- [ ] **Step 3: Types** — in `frontend/src/types/api.ts` replace lines 40–41 (`PPE_LABEL`, `PPEItemInfo`) with:

```ts
export const PPE_LABEL: Record<string, string> = {
  hard_hat: "Hard hat", safety_vest: "Safety vest", gloves: "Gloves", goggles: "Goggles", safety_shoes: "Safety shoes",
  fall: "Fall", fight: "Fighting", phone: "Phone use", stalling: "Stalling work",
};
export const BEHAVIOUR_IDS = new Set(["fall", "fight", "phone", "stalling"]);
export interface PPEItemInfo { id: string; label: string; region: "head" | "eyes" | "torso" | "hands" | "feet" | null; kind: "ppe" | "behaviour" }
export interface Schedule { tz: string; days: number[]; windows: string[] }
```

In `interface Rule`, change the first line to
`id: string; name: string; ppe_class: string; camera_id: string | null; zone_id: string | null; schedule: Schedule | null;`.

`frontend/src/services/api.ts`: the `patchRule` body type gains `schedule?: Schedule | null;` (import `Schedule` with the other types).

- [ ] **Step 4: Admin** — `frontend/src/features/Admin.tsx`:

4a. Add `Schedule` to the type import on line 6.

4b. `EMPTY_RULE`: add `schedule: null as Schedule | null` (createRule's body type is `Omit<Rule, ...>`, which now includes `schedule`).

4c. Add above `export function RulesSettings()`:

```tsx
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
export const DEFAULT_SCHEDULE: Schedule = { tz: "Asia/Dubai", days: [0, 1, 2, 3, 4], windows: ["07:00-12:00", "13:00-17:00"] };

/** Shift hours a rule is enforced in (site-local time). Breaks = the gaps between windows. */
export function ScheduleEditor({ value, onChange }: { value: Schedule | null; onChange: (s: Schedule | null) => void }) {
  return (
    <fieldset className="form-grid">
      <legend>Schedule</legend>
      <label><input type="checkbox" checked={value !== null} onChange={(e) => onChange(e.target.checked ? DEFAULT_SCHEDULE : null)} /> only during shift hours <small className="muted">unchecked = always enforced</small></label>
      {value && (
        <>
          <label>Time zone (IANA, e.g. Asia/Dubai, Asia/Riyadh)<input value={value.tz} onChange={(e) => onChange({ ...value, tz: e.target.value })} /></label>
          {/* committed on blur: parsing on every keystroke would swallow the space between windows */}
          <label>Shift windows, breaks left out (e.g. 07:00-12:00 13:00-17:00)<input key={value.windows.join(" ")} defaultValue={value.windows.join(" ")} onBlur={(e) => onChange({ ...value, windows: e.target.value.trim().split(/\s+/).filter(Boolean) })} /></label>
          <div role="group" aria-label="Days">
            {DAYS.map((d, i) => (
              <label key={d}><input type="checkbox" checked={value.days.includes(i)} onChange={(e) => onChange({ ...value, days: e.target.checked ? [...value.days, i].sort((a, b) => a - b) : value.days.filter((x) => x !== i) })} />{d}</label>
            ))}
          </div>
        </>
      )}
    </fieldset>
  );
}
```

4d. In the rule card, after the temporal-policy inputs' closing `</div>` and before the Save button, add:
`<ScheduleEditor value={r.schedule} onChange={(s) => upd(r.id, (x) => ({ ...x, schedule: s }))} />`
and in `save` add `schedule: r.schedule` to the `patchRule` body.

4e. New-rule form: replace the `Item` select with grouped options:

```tsx
        <label>Check<select value={ruleForm.ppe_class} onChange={(e) => setRuleForm({ ...ruleForm, ppe_class: e.target.value })}>
          <optgroup label="PPE">{items.filter((i) => i.kind !== "behaviour").map((i) => <option key={i.id} value={i.id}>{i.label}</option>)}</optgroup>
          <optgroup label="Behaviour">{items.filter((i) => i.kind === "behaviour").map((i) => <option key={i.id} value={i.id}>{i.label}</option>)}</optgroup>
        </select></label>
```

- [ ] **Step 5: Live overlay and Events labels**

`frontend/src/components/LiveView.tsx` — below `ITEM_SHORT` add
`const BEHAVIOUR_SHORT: Record<string, string> = { fall: "PERSON DOWN", fight: "FIGHT", phone: "PHONE", stalling: "STALLING" };`
and in the label line replace `` `NO ${ITEM_SHORT[t.ppe_class] ?? t.ppe_class.toUpperCase()}` `` with
`` (BEHAVIOUR_SHORT[t.ppe_class] ?? `NO ${ITEM_SHORT[t.ppe_class] ?? t.ppe_class.toUpperCase()}`) ``.

`frontend/src/features/Events.tsx` — import `BEHAVIOUR_IDS` from `../types/api`; in the Votes `<dd>` replace the literal word `without` with
`{BEHAVIOUR_IDS.has(detail.confidence_summary.ppe_class ?? "") ? "showing" : "without"}`.

- [ ] **Step 6: Run** — `cd frontend && npm run lint && npx vitest run && npm run build` → all green.

- [ ] **Step 7: Verify in the browser** — run the app (preview tools), open Rules & settings: the Check select shows a Behaviour group; create a `phone` rule; tick "only during shift hours", edit windows, Save; reload the page and confirm the schedule persisted. Screenshot for the PR.

- [ ] **Step 8: Commit**

```bash
git add frontend/src
git commit -m "feat(phase4): behaviour checks and shift schedule editor in the UI"
```

---

### Task 8: Measure fall detection on real accident footage (OmniFall)

**Files:**
- Create: `scripts/ml/omnifall_to_csv.py`, `scripts/ml/eval_fall.py`
- Output: `docs/ml/fall_eval_itw_fps5.json`, `docs/ml/fall_eval_itw_fps10.json` (and `_syn_` variants)

**Interfaces:**
- Consumes: `FallAssessor`, `BEHAVIOURS["fall"].policy`, `PPEStateMachine`, `make_tracker`, `Detection`, `Frame` — the production code path minus RTSP and DB.
- Produces: JSON reports with `falls, tp, fn, fp, precision, recall, f1, false_alarms_per_hour, missed_seen_lying_without_event, fps`.

- [ ] **Step 1: Label export script** — create `scripts/ml/omnifall_to_csv.py`:

```python
"""Export an OmniFall split to the eval_fall.py CSV (video,start,end,label).

Labels are CC BY-NC-SA 4.0 (evaluation only, never redistributed). Run in a throwaway env, never as a backend dep:
  uv run --with omnifall --with datasets python scripts/ml/omnifall_to_csv.py --config of-itw --out data/fall/of_itw_test.csv
OF-ItW needs the one-time `omnifall prepare-oops` (~45 GB streamed, ~2.6 GB kept); OF-Syn videos auto-download.
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

    ds = omnifall.add_video(load_dataset("simplexsigil2/omnifall", a.config, split=a.split), config=a.config)
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
```

- [ ] **Step 2: Evaluation script** — create `scripts/ml/eval_fall.py`:

```python
"""Event-level fall evaluation (Phase 4, ADR-015): pose model + ByteTrack + FallAssessor + the fall rule's state
machine run offline on labelled videos - the production code path minus RTSP and the database.

  python scripts/ml/eval_fall.py --labels data/fall/of_itw_test.csv --fps 5 --report docs/ml/fall_eval_itw_fps5.json
Labels CSV: video,start,end,label (seconds). Rows labelled 'fall' are fall events; every video in the CSV is
processed. A fall counts as detected when the rule CONFIRMS within [start, end + hold + 5 s]; every other
confirmation is a false alarm (also per hour of processed footage). 'missed_seen_lying_without_event' counts missed
falls where a lying person was seen without a fall event (ID switch mid-fall, or a fall slower than fall_window_s).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def run_video(path: str, fps: float, pose, args):
    import cv2

    from app.behaviour.assessors import FallAssessor
    from app.config.settings import BEHAVIOURS, BehaviourConfig, RuleThresholds, TrackingConfig
    from app.inference.types import Detection
    from app.ppe.state_machine import PPEStateMachine
    from app.ppe.types import PPEState
    from app.streaming.frame import Frame
    from app.tracking.tracker import make_tracker

    tcfg = TrackingConfig()
    tracker, fall = make_tracker(tcfg, fps), FallAssessor(BehaviourConfig(), RuleThresholds())
    fsm = PPEStateMachine("eval", BEHAVIOURS["fall"].policy, tcfg.track_expiry_s)
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    confirms, lying, i, next_t, t = [], [], 0, 0.0, 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t, i = i / vfps, i + 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / fps
        r = pose.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.data.cpu().numpy(),
                               strict=True):
                dets.append(Detection("person", float(c), tuple(map(float, b)), "eval", k))
        frame = Frame("eval", i, img, None, t, t)
        obs = fall.assess_deferred(tracker.update(dets, frame), [], frame)
        lying += [t for o in obs if o.reason == "horizontal_without_fall"]
        confirms += [tr.ts for tr in fsm.update(obs, t) + fsm.expire(t) if tr.new is PPEState.CONFIRMED_NON_COMPLIANT]
    cap.release()
    return confirms, lying, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo26s-pose.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="first N videos only (smoke runs)")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    from app.config.settings import BEHAVIOURS

    hold = BEHAVIOURS["fall"].policy.min_duration_s
    falls: dict[str, list[tuple[float, float]]] = {}
    with args.labels.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            falls.setdefault(row["video"], [])  # every video is processed, with or without a fall
            if row["label"] == "fall":
                falls[row["video"]].append((float(row["start"]), float(row["end"])))
    videos = sorted(falls)[: args.limit or None]
    pose = YOLO(args.weights)
    tp = fn = fp = seen_lying = 0
    seconds = 0.0
    for n, v in enumerate(videos, 1):
        confirms, lying, dur = run_video(v, args.fps, pose, args)
        used: set[float] = set()
        for s, e in falls[v]:
            hit = next((c for c in confirms if s <= c <= e + hold + 5.0 and c not in used), None)
            if hit is None:
                fn += 1
                seen_lying += any(s <= x <= e + hold + 5.0 for x in lying)
            else:
                used.add(hit)
                tp += 1
        fp += sum(1 for c in confirms if c not in used)
        seconds += dur
        if n % 50 == 0:
            print(f"{n}/{len(videos)} videos  tp={tp} fn={fn} fp={fp}", flush=True)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {"labels": str(args.labels), "videos": len(videos), "falls": tp + fn, "tp": tp, "fn": fn, "fp": fp,
           "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
           "hours": round(seconds / 3600, 3), "false_alarms_per_hour": round(fp / max(seconds / 3600, 1e-9), 2),
           "missed_seen_lying_without_event": seen_lying, "fps": args.fps, "weights": Path(args.weights).name,
           "hold_s": hold}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run** (2 videos) — `uv run --project backend python scripts/ml/eval_fall.py --labels data/fall/of_itw_test.csv --limit 2 --report docs/benchmarks/scratch/fall_smoke.json` → prints a JSON report, no exception.

- [ ] **Step 4: Full runs (GPU, long)** — run in the background, 5 and 10 fps, OF-ItW test and OF-Syn test:

```bash
uv run --project backend python scripts/ml/eval_fall.py --labels data/fall/of_itw_test.csv --fps 5 --report docs/ml/fall_eval_itw_fps5.json
uv run --project backend python scripts/ml/eval_fall.py --labels data/fall/of_itw_test.csv --fps 10 --report docs/ml/fall_eval_itw_fps10.json
uv run --project backend python scripts/ml/eval_fall.py --labels data/fall/of_syn_test.csv --fps 5 --report docs/ml/fall_eval_syn_fps5.json
```

Decision rule (write the outcome into `docs/ml/phase4-log.md`, Task 11): if `missed_seen_lying_without_event` is ≥ 25 % of `fn`, track ID switches mid-fall are a measured problem — open a follow-up for track stitching (inherit fall history from a track lost < 1 s earlier at the same place); do **not** build it in this phase without that number. Compare against OmniFall's published in-the-wild Fall F1 (best ~65.6, VideoMAE) and say plainly where we land.

- [ ] **Step 5: Commit**

```bash
git add scripts/ml/omnifall_to_csv.py scripts/ml/eval_fall.py docs/ml/fall_eval_*.json
git commit -m "feat(phase4): event-level fall evaluation on OmniFall (in-the-wild + synthetic)"
```

---

### Task 9: Train and measure the fight model (RLVS train, SCFD cross-dataset test)

**Files:**
- Create: `scripts/ml/extract_fight_sequences.py`, `scripts/ml/train_fight.py`
- Output: `models/fight_gru.pt` (git-ignored, like all weights), `docs/ml/fight_eval.json`

**Interfaces:**
- Consumes: `PairSequencer`, `FightGRU`, `SEQ_LEN`, `HZ`, `N_FEAT` (Task 5); `gate`, `RuleThresholds`, `BehaviourConfig`, `make_tracker`.
- Produces: `.npz` with `X (N, T, 102) float32`, `clip_idx (N,) int32`, `clip_names (C,)`, `clip_labels (C,) int8`; checkpoint dict `{state_dict, hidden, seq_len, hz}` loadable by `FightModel`.

**Data (manual, licences recorded in Task 11):**
- RLVS — Kaggle `mohamedmustafa/real-life-violence-situations-dataset` (folders `Violence/`, `NonViolence/`, 2 000 clips) → `data/fight/rlvs/`. Licence: "© original authors" — MVP evaluation only.
- SCFD — `git clone https://github.com/seymanurakti/fight-detection-surv-dataset data/fight/scfd` (folders `fight/`, `noFight/`, 300 surveillance clips). Repo MIT; videos from YouTube.
- RWF-2000 is not used (non-commercial; files currently withheld by its authors).

- [ ] **Step 1: Extraction script** — create `scripts/ml/extract_fight_sequences.py`:

```python
"""Pair keypoint sequences for the fight model (Phase 4, ADR-015), built with the PRODUCTION code:
pose model -> ByteTrack -> person gate -> PairSequencer (app/behaviour/fight.py). One label per clip.

  python scripts/ml/extract_fight_sequences.py --root data/fight/rlvs --pos Violence --neg NonViolence --out data/fight/rlvs.npz
  python scripts/ml/extract_fight_sequences.py --root data/fight/scfd --pos fight --neg noFight --out data/fight/scfd.npz
--fps is the simulated camera processing rate (>= the model's 5 Hz); sequences are kept every --stride seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".mpg"}


def clip_sequences(path: Path, pose, args) -> list[np.ndarray]:
    import cv2

    from app.behaviour.assessors import gate
    from app.behaviour.fight import HZ, SEQ_LEN, PairSequencer
    from app.config.settings import BehaviourConfig, RuleThresholds, TrackingConfig
    from app.inference.types import Detection
    from app.streaming.frame import Frame
    from app.tracking.tracker import make_tracker

    th, tracker = RuleThresholds(), make_tracker(TrackingConfig(), args.fps)
    seq = PairSequencer(SEQ_LEN, HZ, BehaviourConfig().fight_pair_dist)
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, last, i, next_t = [], {}, 0, 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t, i = i / vfps, i + 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / args.fps
        r = pose.predict(img, imgsz=640, conf=0.3, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.data.cpu().numpy(),
                               strict=True):
                dets.append(Detection("person", float(c), tuple(map(float, b)), "x", k))
        persons = [p for p in tracker.update(dets, Frame("x", i, img, None, t, t)) if gate(p, th) is None]
        for pair, s in seq.update(persons, t)[1]:
            if t - last.get(pair, -1e9) >= args.stride:
                last[pair] = t
                out.append(s)
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=0.5)
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo26s-pose.pt"))
    args = ap.parse_args()
    from ultralytics import YOLO

    from app.behaviour.fight import HZ, N_FEAT, SEQ_LEN

    pose = YOLO(args.weights)
    xs, idx, names, labels = [], [], [], []
    for label, sub in ((1, args.pos), (0, args.neg)):
        for p in sorted(q for q in (args.root / sub).rglob("*") if q.suffix.lower() in VIDEO_EXT):
            seqs = clip_sequences(p, pose, args)
            names.append(str(p.relative_to(args.root)))
            labels.append(label)
            xs += seqs
            idx += [len(names) - 1] * len(seqs)
            if len(names) % 100 == 0:
                print(f"{len(names)} clips, {len(xs)} sequences", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack(xs).astype(np.float32) if xs else np.zeros((0, SEQ_LEN, N_FEAT), np.float32)
    np.savez_compressed(args.out, X=X, clip_idx=np.array(idx, np.int32), clip_names=np.array(names),
                        clip_labels=np.array(labels, np.int8), seq_len=SEQ_LEN, hz=HZ)
    print(f"{len(names)} clips ({sum(labels)} fight), {len(xs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Training / evaluation script** — create `scripts/ml/train_fight.py`:

```python
"""Train / evaluate the Phase 4 fight model (ADR-015) on sequences from extract_fight_sequences.py.

  python scripts/ml/train_fight.py train --data data/fight/rlvs.npz --out models/fight_gru.pt --report docs/ml/fight_eval.json
  python scripts/ml/train_fight.py eval --data data/fight/scfd.npz --weights models/fight_gru.pt --report docs/ml/fight_eval.json
Clip-level metrics: a clip is a fight if any of its pair sequences scores >= threshold; clips without any pair
sequence count as 'no fight'. Train/val split is by clip (no leakage). Reports are merged into one JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def clip_scores(probs: np.ndarray, clip_idx: np.ndarray, n_clips: int) -> np.ndarray:
    s = np.zeros(n_clips, np.float32)
    np.maximum.at(s, clip_idx, probs)
    return s


def metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred, y = scores >= thr, labels == 1
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": round(float(thr), 3), "clips": len(labels),
            "accuracy": round((tp + tn) / max(len(labels), 1), 4), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def predict(net, X: np.ndarray, device) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + 1024]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def merge_report(path: Path, key: str, value: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({key: value}, indent=2))


def train(a) -> None:
    import torch

    from app.behaviour.fight import FightGRU

    d = np.load(a.data)
    X, ci, labels = d["X"], d["clip_idx"], d["clip_labels"].astype(np.int64)
    rng = np.random.default_rng(0)
    val = np.zeros(len(labels), bool)
    for c in (0, 1):  # stratified 80/20 split by clip
        ids = rng.permutation(np.flatnonzero(labels == c))
        val[ids[: len(ids) // 5]] = True
    tr_m, va_m = ~val[ci], val[ci]
    y = labels[ci].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FightGRU(a.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = max(float(y[tr_m].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((tr_m.sum() - pos) / pos, device=device))
    Xt, yt = torch.from_numpy(X[tr_m]), torch.from_numpy(y[tr_m])
    best, best_state = -1.0, None
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()
        net.eval()
        vs = clip_scores(predict(net, X[va_m], device), np.searchsorted(np.flatnonzero(val), ci[va_m]), int(val.sum()))
        f1 = metrics(vs, labels[val], 0.5)["f1"]
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  val clip F1@0.5 {f1:.4f}", flush=True)
    net.load_state_dict(best_state)
    vs = clip_scores(predict(net, X[va_m], device), np.searchsorted(np.flatnonzero(val), ci[va_m]), int(val.sum()))
    sweep = [metrics(vs, labels[val], t) for t in np.arange(0.3, 0.95, 0.05)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hidden": a.hidden, "seq_len": int(d["seq_len"]), "hz": float(d["hz"])},
               a.out)
    merge_report(a.report, f"val:{Path(a.data).stem}", {"at_0.5": metrics(vs, labels[val], 0.5),
                                                       "best_f1": max(sweep, key=lambda m: m["f1"]),
                                                       "sequences_train": int(tr_m.sum()), "epochs": a.epochs})


def evaluate(a) -> None:
    import torch

    from app.behaviour.fight import FightModel

    d, m = np.load(a.data), FightModel(a.weights)
    scores = clip_scores(m.score(d["X"]) if len(d["X"]) else np.zeros(0, np.float32), d["clip_idx"],
                         len(d["clip_labels"]))
    labels = d["clip_labels"].astype(np.int64)
    merge_report(a.report, f"test:{Path(a.data).stem}", {"at_0.5": metrics(scores, labels, 0.5),
                                                        "at_0.7_rule_default": metrics(scores, labels, 0.7),
                                                        "weights": m.version, "torch": torch.__version__})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=256)
    e = sub.add_parser("eval")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--weights", type=Path, required=True)
    for p in (t, e):
        p.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run** (GPU, background; each extraction is long):

```bash
uv run --project backend python scripts/ml/extract_fight_sequences.py --root data/fight/rlvs --pos Violence --neg NonViolence --out data/fight/rlvs.npz
uv run --project backend python scripts/ml/extract_fight_sequences.py --root data/fight/scfd --pos fight --neg noFight --out data/fight/scfd.npz
uv run --project backend python scripts/ml/train_fight.py train --data data/fight/rlvs.npz --out models/fight_gru.pt --report docs/ml/fight_eval.json
uv run --project backend python scripts/ml/train_fight.py eval --data data/fight/scfd.npz --weights models/fight_gru.pt --report docs/ml/fight_eval.json
```

Expected: a JSON with `val:rlvs` and `test:scfd`. Literature reference points for honest comparison: skeleton models 93 % in-dataset on RWF-2000; 83–88 % cross-dataset (RGB X3D). If `best_f1.threshold` differs from 0.7, note it — the operator sets it per rule as `absent_conf`.

- [ ] **Step 4: Enable for the demo** — add `fight_weights: fight_gru.pt` to the local `configs/models.yaml` (not the example) and restart; check the worker log lists `fight` in model versions.

- [ ] **Step 5: Commit** (scripts + report only; weights stay out of git)

```bash
git add scripts/ml/extract_fight_sequences.py scripts/ml/train_fight.py docs/ml/fight_eval.json
git commit -m "feat(phase4): fight model training and cross-dataset evaluation (RLVS -> SCFD)"
```

---

### Task 10: Phone detection quality and the GPU cost of Phase 4

**Files:**
- Create: `scripts/ml/eval_phone_coco.py`
- Modify: `scripts/benchmark/run_benchmark.py`
- Output: `docs/ml/phone_coco_eval.json`, `docs/benchmarks/results/p4_uhd_cams10_fps5_nvdec.json`

- [ ] **Step 1: COCO phone evaluation script** — create `scripts/ml/eval_phone_coco.py`:

```python
"""COCO val2017 'cell phone' (class 67) quality of the phone-stage weights at the deployed input sizes.

  python scripts/ml/eval_phone_coco.py --weights models/yolo26s.pt --imgsz 640 1280 --report docs/ml/phone_coco_eval.json
Downloads val2017 images (~1 GB) + labels into data/coco on first run. A PROXY only: COCO phones are mostly
close-up; CCTV-distance phones are smaller - the field number must come from site footage (docs/ml/phase4-log.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHONE = 67


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo26s.pt"))
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.downloads import download

    root = a.data_root
    if not (root / "images" / "val2017").exists():
        download(["https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"], dir=root.parent)
        download(["http://images.cocodataset.org/zips/val2017.zip"], dir=root / "images")
    model = YOLO(a.weights)
    cfg = root / "val_only.yaml"
    cfg.write_text(yaml.safe_dump({"path": str(root), "train": "images/val2017", "val": "images/val2017",
                                   "names": model.names}), encoding="utf-8")
    out = []
    for imgsz in a.imgsz:
        r = model.val(data=str(cfg), imgsz=imgsz, batch=16, half=True, plots=False, verbose=False)
        i = list(r.box.ap_class_index).index(PHONE)
        out.append({"imgsz": imgsz, "precision": round(float(r.box.p[i]), 4), "recall": round(float(r.box.r[i]), 4),
                    "ap50": round(float(r.box.ap50[i]), 4), "ap50_95": round(float(r.box.ap[i]), 4)})
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"weights": Path(a.weights).name, "class": "cell phone", "results": out},
                                   indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

Run: `uv run --project backend python scripts/ml/eval_phone_coco.py --report docs/ml/phone_coco_eval.json`. If the label zip layout differs (labels must end up in `data/coco/labels/val2017/`), move them there and rerun — the yaml expects the standard Ultralytics layout.

- [ ] **Step 2: Benchmark profile** — in `scripts/benchmark/run_benchmark.py`:
  - add to `PROFILES`: `"p4_uhd_cams10_fps5_nvdec": {"cams": 10, "fps": 5, "rules": "construction", "items": True, "hwaccel": "cuda", "phone": True, "behaviours": True},`
  - below `ITEMS_MODEL`: `PHONE_MODEL = {"weights": "yolo26s.pt", "imgsz": 1280, "conf": 0.25}  # Phase 4 phone stage`
  - change the settings import in `run_profile` to `from app.config.settings import (ModelSpec, RuleSeed, Settings, load_app_config, load_camera_seeds, load_models_config, load_rule_seeds)` and delete the two inner `from app.config.settings import ...` lines;
  - after the preset block: `if p.get("behaviours"): rules += [RuleSeed(id=f"{b}-check", name=b, ppe_class=b) for b in ("fall", "fight", "phone", "stalling")]`
  - after the `items` block: `if p.get("phone"): models.phone = ModelSpec(**PHONE_MODEL)`

- [ ] **Step 3: Run** — start the 4K simulator as for Phase 3 (`docs/benchmarks/phase3-4k.md`), stop the API, then
`uv run --project backend python scripts/benchmark/run_benchmark.py --profile p4_uhd_cams10_fps5_nvdec`.
Compare processed FPS, p95 frame age and GPU % against `p3_uhd_cams10_fps5_nvdec`; the delta is the measured cost of the phone stage + four behaviour rules.

- [ ] **Step 4: Commit**

```bash
git add scripts/ml/eval_phone_coco.py scripts/benchmark/run_benchmark.py docs/ml/phone_coco_eval.json docs/benchmarks/results/p4_uhd_cams10_fps5_nvdec.json
git commit -m "feat(phase4): phone detection proxy eval and Phase 4 GPU benchmark profile"
```

---

### Task 11: Documentation and release 0.4.0

**Files:**
- Create: `docs/decisions/ADR-015-behaviour-checks.md`, `docs/ml/phase4-log.md`
- Modify: `README.md`, `CHANGELOG.md`, `BUILD_STATUS.md`, `docs/model-and-data-licenses.md`, `docs/system-design/10-api-and-websocket-contracts.md`, `docs/system-design/12-security-and-privacy-boundaries.md`, `docs/testing/test-report.md`, `docs/benchmarks/README.md`

Write numbers **only** from the JSON files produced in Tasks 8–10; where a run was not done, say "not measured".

- [ ] **Step 1: ADR-015** (follow `docs/decisions/ADR-000-template.md`): context (user scope 2026-11-20; Gulf deployment), decision (behaviours as rule checks reusing the PPE state machine; fall = fast upright→lying sequence + duration hold; stalling = keypoint stillness; phone = COCO class 67 over hand/head ROI with gap bridging; fight = pair-keypoint GRU, CPU, train/serve shared features; shift schedule per rule), rejected options (angry: emotion inference, unreliable from CCTV (~59–61 % anger accuracy on close-up faces, AffectNet) and dropped by the user; "not working": undefined; separate behaviour pipeline: duplicates the event path; RGB video transformer for fights: GPU cost, no measured need yet; track stitching: wait for the Task 8 ID-switch number), consequences (UNCERTAIN without pose; fight needs target_fps ≥ 5; stalling measures stillness not intent; teacher stays PPE-only).

- [ ] **Step 2: `docs/ml/phase4-log.md`**: one section per check with the measured table (fall: OF-ItW/OF-Syn × 5/10 fps — precision, recall, F1, false alarms/hour, missed-while-lying count and the track-stitching decision; fight: RLVS val + SCFD cross-dataset, best threshold; phone: COCO AP at 640/1280 + the proxy caveat; stalling: "timer correctness only - no public ground truth"; GPU cost delta from the benchmark). Include the literature reference points cited in the research (OmniFall wild best ~65.6 F1; RWF-2000 skeleton ~93 %; cross-dataset 83–88 %).

- [ ] **Step 3: Licences** — `docs/model-and-data-licenses.md`: OmniFall annotations CC BY-NC-SA 4.0 (eval only, OOPS/synthetic videos from source); RLVS "© original authors" (train, MVP only); SCFD MIT repo, YouTube-sourced videos (test); COCO val2017 (annotations CC BY 4.0, Flickr image terms; eval only); `yolo26s.pt` AGPL-3.0 (same as the existing stack); **fight_gru.pt: MVP weights, must be retrained on licensed or client data before commercial deployment.**

- [ ] **Step 4: Privacy** — `12-security-and-privacy-boundaries.md`, new "Phase 4 behaviour checks" section: no identity is stored (track ids are per run); evidence is the privacy-masked frame as before; Gulf deployment checklist: UAE PDPL (Federal Decree-Law 45/2021) / KSA PDPL — written employee notice and signage, stated purpose (safety and shift discipline), legal basis confirmed by the client's counsel (consent vs employment exception differs by source), proportional retention (`retention_days`), DIFC/ADGM free zones have their own law; stalling and phone rules should be scoped by zone + schedule.

- [ ] **Step 5: Contracts and status** — API doc: `/ppe-items` `kind`, rule `schedule`; README: Phase 4 feature list + how to enable phone/fight weights; CHANGELOG `## 0.4.0 - <date>` (link this plan and ADR-015); BUILD_STATUS; test-report (new test counts from `uv run pytest tests -q -m "not gpu"` and vitest); benchmarks README row for `p4_*`.

- [ ] **Step 6: Final verification** — `cd backend && uv run ruff check . && uv run pytest tests -q -m "not gpu"`; `cd frontend && npm run lint && npx vitest run && npm run build`. All green before committing.

- [ ] **Step 7: Commit**

```bash
git add docs README.md CHANGELOG.md BUILD_STATUS.md
git commit -m "docs(phase4): ADR-015, measured results, licences, privacy checklist, 0.4.0"
```

---

## Self-review notes (planner)

- Spec coverage: fall with temporal confirmation (Tasks 3, 6, 8); fighting (5, 6, 9); phone in work hours (4, 6 + schedule from 1–2, 10); stalling (3, 6); normal = no event (implicit, PRESENT outcomes); everything editable in the UI (rules, hold time via min_duration_s, schedule, zones, absent_conf) (2, 7); angry / not working dropped and accident detection parked (Global Constraints, ADR-015).
- Out of scope, named: track stitching for ID switches mid-fall (only if Task 8 shows ≥ 25 % of misses), skipping the phone GPU stage when no phone rule is enabled, teacher review of behaviour clips, a Gulf preset file (operators can create presets in the UI), multi-person fight event merging beyond the existing spatial dedupe.
- Known risks: pose keypoints degrade on lying people; RLVS/SCFD are YouTube footage, not CCTV-from-above; COCO phone AP is a proxy; stalling has no accuracy figure by design.
