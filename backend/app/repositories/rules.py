from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import RuleSeed, ZoneSeed
from app.models import PPERule, Zone


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
