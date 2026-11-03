"""Shared test fixtures. Integration tests need `docker compose up -d` (PostgreSQL, MediaMTX, Toxiproxy)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_dotenv() -> dict[str, str]:
    env = {}
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                env[k.strip()] = v.strip()
    return env


@pytest.fixture(scope="session")
def dotenv() -> dict[str, str]:
    env = load_dotenv()
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


@pytest.fixture(scope="session")
def streams_dir(dotenv) -> Path:
    d = Path(dotenv.get("STREAMS_DIR", REPO / "data" / "streams"))
    if not d.is_absolute():
        d = REPO / d
    if not (d / "cam02.mp4").exists():
        pytest.skip(f"demo clips missing in {d} (run scripts/demo/fetch_demo_videos.py)")
    return d
