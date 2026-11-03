"""Programmatic Alembic upgrade (run at API start, NFR-REL-004)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

BACKEND = Path(__file__).resolve().parents[2]


def alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.attributes["url"] = url  # read by env.py; never written to the ini (it contains the password)
    return cfg


def upgrade_to_head(url: str) -> None:
    command.upgrade(alembic_config(url), "head")
