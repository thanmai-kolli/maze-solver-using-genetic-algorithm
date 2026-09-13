"""Capture README screenshots of the Streamlit UI.

Starts the app on a spare port, drives it with Playwright and writes PNGs to
docs/images/. Requires: pip install playwright && playwright install chromium

    python scripts/make_screenshots.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
VIEWPORT = {"width": 1680, "height": 1040}
SCALE = 2


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_app(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port), "--server.headless", "true"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_ready(page: Page) -> None:
    page.wait_for_selector("text=MazeGA", timeout=60_000)
    page.wait_for_timeout(2500)
    page.add_style_tag(content='[data-testid="stToolbar"],[data-testid="stDecoration"]'
                               "{display:none!important}")


def shot(page: Page, name: str) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(900)
    page.screenshot(path=str(OUT / name))
    print("wrote", OUT / name)


def choose_maze(page: Page) -> None:
    """Switch to a 25x25 maze that takes the GA ~85 generations to crack."""
    slider = page.get_by_role("slider", name="Size (cells per side)").first
    slider.evaluate("el => el.focus()")
    for _ in range(2):
        slider.press("ArrowRight")
        page.wait_for_timeout(500)
    seed = page.locator('[data-testid="stSidebar"] [data-testid="stNumberInput"] input').first
    seed.fill("2")
    seed.press("Enter")
    page.wait_for_timeout(1200)
    page.get_by_role("button", name="Generate maze").click()
    page.wait_for_timeout(2500)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = start_app(port)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
            for _ in range(30):
                try:
                    page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
                    break
                except Exception:
                    time.sleep(1)
            wait_ready(page)
            choose_maze(page)

            shot(page, "ui-build.png")

            page.get_by_role("tab", name="Evolve").click()
            page.wait_for_timeout(800)
            page.get_by_role("button", name="Run evolution").click()
            page.wait_for_selector("text=Replay generation", timeout=300_000)
            page.wait_for_timeout(4000)
            shot(page, "ui-evolve.png")

            page.get_by_role("tab", name="Analytics").click()
            page.wait_for_timeout(4000)
            shot(page, "ui-analytics.png")

            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
