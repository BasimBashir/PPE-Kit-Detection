"""Create .env from .env.example, filling every empty *_PASSWORD / *_PASS value with a random secret.

Existing .env is never overwritten. Usage: python scripts/bootstrap/init_env.py [--streams-dir DIR]
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams-dir", help="override STREAMS_DIR")
    args = ap.parse_args()
    env = ROOT / ".env"
    if env.exists():
        print(".env already exists; leaving it untouched")
        return 0
    out = []
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        key, sep, val = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            if not val and key.endswith(("_PASSWORD", "_PASS")):
                val = secrets.token_urlsafe(18)
            if key == "STREAMS_DIR" and args.streams_dir:
                val = args.streams_dir
            line = f"{key}={val}"
        out.append(line)
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {env} (random secrets generated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
