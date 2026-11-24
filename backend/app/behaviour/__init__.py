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
