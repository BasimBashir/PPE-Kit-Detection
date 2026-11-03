"""Capture dashboard screenshots from the running app with the system Edge/Chrome (no browser download):

  uv run --with playwright python scripts/demo/screenshots.py [--out docs/demo/screenshots] [--channel msedge]
Logs in with the admin password from .env, visits each page and saves full-page PNGs. Doubles as a smoke test:
exits non-zero if a page shows an error banner or no camera cards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PAGES = [("overview", "/"), ("camera-wall", "/cameras"), ("camera-detail-cam05", "/cameras/cam05"),
         ("events", "/events"), ("system-health", "/health"), ("rules", "/config/rules"),
         ("cameras-config", "/config/cameras"), ("demo-control", "/demo")]


def admin_password() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("PPE_ADMIN_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("PPE_ADMIN_PASSWORD missing in .env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo" / "screenshots"))
    ap.add_argument("--channel", default="msedge")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel=a.channel, headless=True)
        page = b.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(a.base + "/login")
        page.get_by_label("Username").fill("admin")
        page.get_by_label("Password").fill(admin_password())
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_url(a.base + "/")
        page.wait_for_selector(".tile-value", timeout=15000)
        for name, path in PAGES:
            page.goto(a.base + path)
            page.wait_for_timeout(3500)  # live previews / websocket snapshot
            if page.locator(".error[role=alert]").count():
                problems.append(f"{name}: {page.locator('.error[role=alert]').first.inner_text()}")
            page.screenshot(path=str(out / f"{name}.png"), full_page=True)
            print("saved", name, flush=True)
        if page.goto(a.base + "/cameras") and page.wait_for_timeout(2000) is None \
                and page.locator(".cam-card").count() == 0:
            problems.append("camera wall shows no camera cards")
        b.close()
    for pr in problems:
        print("PROBLEM:", pr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
