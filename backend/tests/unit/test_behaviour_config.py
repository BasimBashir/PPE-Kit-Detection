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
