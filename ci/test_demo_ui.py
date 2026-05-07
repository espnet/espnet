from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

SYSTEM_CONFIGS = {
    "default_asr": {
        "demo_dir": "egs3/mini_an4/asr/exp/demo_ui_default",
        "expected_texts": ["Input Audio", "Transcription"],
        "run_label": "Run",
    },
    "custom_asr_image": {
        "demo_dir": "egs3/mini_an4/asr/exp/demo_ui_custom",
        "expected_texts": ["Input Audio", "Transcription", "Reference Image"],
        "run_label": "Run",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wait_for_url(proc: subprocess.Popen, port: int, timeout: float = 60.0) -> str:
    pattern = re.compile(rf"(https?://127\.0\.0\.1:{port})")
    deadline = time.time() + timeout
    stdout = proc.stdout
    if stdout is None:
        raise RuntimeError("demo app stdout is not captured.")
    while time.time() < deadline:
        ready, _, _ = select.select([stdout], [], [], 0.2)
        if ready:
            line = stdout.readline()
            if not line:
                break
            match = pattern.search(line)
            if match:
                return match.group(1)
        if proc.poll() is not None:
            break
    raise RuntimeError("demo app failed to start or emit a local URL.")


def _start_demo_app(demo_dir: Path, *, port: int) -> tuple[subprocess.Popen, str]:
    env = os.environ.copy()
    env["GRADIO_ANALYTICS_ENABLED"] = "0"
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"
    env["GRADIO_SERVER_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(demo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    url = _wait_for_url(proc, port)
    return proc, url


def _stop_demo_app(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.parametrize(
    "system_name,config",
    sorted(SYSTEM_CONFIGS.items()),
)
def test_demo_ui_labels(system_name: str, config: dict) -> None:
    demo_dir = _repo_root() / config["demo_dir"]
    expected_texts = list(config.get("expected_texts", []))

    port = 7860 + list(sorted(SYSTEM_CONFIGS)).index(system_name)
    proc, url = _start_demo_app(demo_dir, port=port)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(1000)
                body_text = page.locator("body").inner_text()
                missing = [text for text in expected_texts if text not in body_text]
                assert not missing, f"{system_name} missing UI text: {missing}"
                run_label = config.get("run_label", "Run")
                page.get_by_role("button", name=re.compile(run_label, re.I)).wait_for()
            finally:
                browser.close()
    finally:
        _stop_demo_app(proc)
