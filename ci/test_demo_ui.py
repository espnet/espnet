from __future__ import annotations

import base64
import os
import re
import select
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

SYSTEM_CONFIGS = {
    "default_asr": {
        "demo_dir": "egs3/mini_an4/asr/exp/demo_ui_default",
        "expected_texts": ["Input Audio", "Transcription"],
        "expected_image_inputs": 0,
        "expected_output": "speech=1",
        "run_label": "Run",
    },
    "custom_asr_image": {
        "demo_dir": "egs3/mini_an4/asr/exp/demo_ui_custom",
        "expected_texts": ["Input Audio", "Transcription", "Reference Image"],
        "expected_image_inputs": 1,
        "expected_output": "speech=1 image=1",
        "run_label": "Run",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wait_for_url(proc: subprocess.Popen, port: int, timeout: float = 60.0) -> str:
    url = f"http://127.0.0.1:{port}"
    output_lines: list[str] = []
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
            output_lines.append(line.rstrip())
        try:
            with urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return url
        except URLError:
            pass
        if proc.poll() is not None:
            break
    output_tail = "\n".join(output_lines[-20:])
    raise RuntimeError(
        "demo app failed to start.\n"
        f"demo_dir={proc.args}\n"
        f"captured_output:\n{output_tail}"
    )


def _start_demo_app(demo_dir: Path, *, port: int) -> tuple[subprocess.Popen, str]:
    env = os.environ.copy()
    env["GRADIO_ANALYTICS_ENABLED"] = "0"
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"
    env["GRADIO_SERVER_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = ""
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


def _write_test_audio(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 1600)


def _write_test_image(path: Path) -> None:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
        "AusB9Wn5V1cAAAAASUVORK5CYII="
    )
    path.write_bytes(png_bytes)


def check_demo_ui_labels(system_name: str, config: dict) -> None:
    demo_dir = _repo_root() / config["demo_dir"]
    expected_texts = list(config.get("expected_texts", []))
    expected_image_inputs = int(config.get("expected_image_inputs", 0))
    expected_output = str(config.get("expected_output", ""))

    port = 7860 + list(sorted(SYSTEM_CONFIGS)).index(system_name)
    proc, url = _start_demo_app(demo_dir, port=port)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                ui_texts = page.evaluate("""() => {
                        const values = new Set();
                        const push = (value) => {
                            if (typeof value !== "string") {
                                return;
                            }
                            const normalized = value.trim();
                            if (normalized) {
                                values.add(normalized);
                            }
                        };
                        for (const element of document.querySelectorAll("*")) {
                            push(element.textContent);
                            push(element.getAttribute("aria-label"));
                            push(element.getAttribute("alt"));
                            push(element.getAttribute("title"));
                        }
                        return Array.from(values);
                    }""")
                missing = [
                    text
                    for text in expected_texts
                    if not any(text in candidate for candidate in ui_texts)
                ]
                assert not missing, f"{system_name} missing UI text: {missing}"
                image_inputs = page.locator(
                    "input[type='file'][accept*='image']"
                ).count()
                assert image_inputs == expected_image_inputs, (
                    f"{system_name} expected {expected_image_inputs} image input(s), "
                    f"but found {image_inputs}"
                )
                run_label = config.get("run_label", "Run")
                run_button = page.get_by_role(
                    "button", name=re.compile(run_label, re.I)
                )
                run_button.wait_for()

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    audio_path = temp_dir / "sample.wav"
                    _write_test_audio(audio_path)
                    page.locator("input[type='file'][accept*='audio']").set_input_files(
                        str(audio_path)
                    )
                    if expected_image_inputs:
                        image_path = temp_dir / "sample.png"
                        _write_test_image(image_path)
                        page.locator(
                            "input[type='file'][accept*='image']"
                        ).set_input_files(str(image_path))
                    run_button.click()

                output_box = page.get_by_role(
                    "textbox", name=re.compile("transcription", re.I)
                )
                output_box.wait_for()
                page.wait_for_function(
                    """(expected) => {
                        const textboxes = Array.from(
                            document.querySelectorAll('textarea, input[type="text"]')
                        );
                        return textboxes.some((node) => node.value.includes(expected));
                    }""",
                    arg=expected_output,
                    timeout=30000,
                )
                output_value = output_box.input_value()
                assert expected_output in output_value, (
                    f"{system_name} expected output '{expected_output}' "
                    f"but got '{output_value}'"
                )
            finally:
                browser.close()
    finally:
        _stop_demo_app(proc)


def main() -> None:
    for system_name, config in sorted(SYSTEM_CONFIGS.items()):
        check_demo_ui_labels(system_name, config)


if __name__ == "__main__":
    main()
