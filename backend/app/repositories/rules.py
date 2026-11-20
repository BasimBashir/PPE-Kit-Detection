from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import PPE_ITEMS, RuleSeed, ZoneSeed
from app.models import PPERule, Setting, Zone


def list_rules(s: Session) -> list[PPERule]:
    return list(s.scalars(select(PPERule).order_by(PPERule.id)))


def get_rule(s: Session, rule_id: str) -> PPERule | None:
    return s.get(PPERule, rule_id)


def upsert_rule(s: Session, seed: RuleSeed) -> PPERule:
    """Validated by RuleSeed (pydantic) before it reaches the database (FR-DET-008)."""
    data = seed.model_dump()
    data["thresholds"] = seed.thresholds.model_dump()
    data["temporal_policy"] = seed.temporal_policy.model_dump()
    rule = s.get(PPERule, seed.id)
    if rule is None:
        rule = PPERule(**data)
        s.add(rule)
    else:
        for k, v in data.items():
            setattr(rule, k, v)
    s.flush()
    return rule


def rule_to_seed(rule: PPERule) -> RuleSeed:
    return RuleSeed.model_validate({c: getattr(rule, c) for c in RuleSeed.model_fields})


# ---------------------------------------------------------------- presets (Phase 3, ADR-014)
# A preset is an operator-defined named rule set: {item: "mandatory" | "secondary"} (items left out = off), stored in
# the settings table; applying it to a scope (all cameras / camera / zone) creates or updates one rule per item.
PRESETS_KEY = "presets"


def get_presets(s: Session) -> dict[str, dict[str, str]]:
    row = s.get(Setting, PRESETS_KEY)
    return dict(row.value) if row and row.value else {}


def _save_presets(s: Session, presets: dict[str, dict[str, str]]) -> None:
    row = s.get(Setting, PRESETS_KEY)
    if row is None:
        s.add(Setting(key=PRESETS_KEY, value=presets))
    else:
        row.value = presets  # new dict object => SQLAlchemy sees the JSONB change
    s.flush()


def put_preset(s: Session, name: str, items: dict[str, str]) -> None:
    _save_presets(s, {**get_presets(s), name: items})


def delete_preset(s: Session, name: str) -> bool:
    p = get_presets(s)
    if p.pop(name, None) is None:
        return False
    _save_presets(s, p)
    return True


def apply_preset(s: Session, name: str, camera_id: str | None, zone_id: str | None) -> list[PPERule]:
    """Idempotent: re-applying updates the same rules (id = <preset>-<item>-<scope>); thresholds/policy edits kept."""
    scope = zone_id or camera_id or "all"
    out = []
    for item, mode in get_presets(s)[name].items():
        rid = f"{name}-{item}-{scope}"[:64]  # shortcut: truncation may collide for >64-char ids; ids are short today
        mandatory = mode == "mandatory"
        old = get_rule(s, rid)
        seed = rule_to_seed(old) if old else RuleSeed(
            id=rid, name=f"{PPE_ITEMS[item].label} ({name})", ppe_class=item, camera_id=camera_id, zone_id=zone_id)
        seed = seed.model_copy(update={"mandatory": mandatory, "enabled": True,
                                       "severity": "high" if mandatory else "low"})
        out.append(upsert_rule(s, seed))
    return out


def seed_presets(s: Session, presets_dir: Path) -> int:
    """First start only: the shipped presets (configs/presets/*.yaml) become editable operator presets."""
    if s.get(Setting, PRESETS_KEY) is not None or not presets_dir.is_dir():
        return 0
    presets = {}
    for f in sorted(presets_dir.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        rules = [RuleSeed.model_validate(r) for r in data.get("rules", [])]
        presets[f.stem] = {r.ppe_class: "mandatory" if r.mandatory else "secondary" for r in rules}
    _save_presets(s, presets)
    return len(presets)


def list_zones(s: Session, camera_id: str | None = None) -> list[Zone]:
    q = select(Zone).order_by(Zone.id)
    if camera_id:
        q = q.where(Zone.camera_id == camera_id)
    return list(s.scalars(q))


def get_zone(s: Session, zone_id: str) -> Zone | None:
    return s.get(Zone, zone_id)


def upsert_zone(s: Session, seed: ZoneSeed) -> Zone:
    data: dict[str, Any] = seed.model_dump()
    data["polygon"] = [list(p) for p in seed.polygon]
    z = s.get(Zone, seed.id)
    if z is None:
        z = Zone(**data)
        s.add(z)
    else:
        for k, v in data.items():
            setattr(z, k, v)
    s.flush()
    return z


def delete_zone(s: Session, z: Zone) -> None:
    s.delete(z)
    s.flush()
