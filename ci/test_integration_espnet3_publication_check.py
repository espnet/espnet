#!/usr/bin/env python3

"""Validate a packed ESPnet3 publication bundle through ``espnet3.api.inference.load``.

The way every front end loads a published model: without trusting bundled
code, from a file path, expecting the contract's ``text``.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from espnet3.api.inference import Audio, InferenceAPI, load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        help="Unused compatibility flag kept for the CI wrapper.",
    )
    parser.add_argument(
        "--recipe-dir",
        default=".",
        help="Unused compatibility flag kept for the CI wrapper.",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Optional remote model tag, checked via espnet3.api.inference.load().",
    )
    return parser.parse_args()


def _download_sample_audio(tmp_dir: Path) -> Path:
    """Download one short WAV file for publication smoke checks."""
    asset_name = "tutorial-assets/steam-train-whistle-daniel_simon.wav"
    sample_path = tmp_dir / Path(asset_name).name
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from torchaudio.utils import download_asset

        downloaded = Path(download_asset(asset_name, path=str(sample_path)))
        if downloaded.is_file():
            return downloaded
    except Exception:
        pass

    sample_url = (
        "https://download.pytorch.org/torchaudio/"
        "tutorial-assets/steam-train-whistle-daniel_simon.wav"
    )
    with urllib.request.urlopen(sample_url, timeout=15) as response:
        sample_path.write_bytes(response.read())
    return sample_path


def _load_sample_audio() -> tuple[dict[str, Any], Path]:
    """Return a minimal ASR-friendly sample dict, and the file it came from."""
    temp_root = Path(
        os.environ.get("TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("TMP")
        or "/tmp"
    )
    sample_path = _download_sample_audio(temp_root / "espnet3-publication-assets")
    speech, _sample_rate = sf.read(str(sample_path), dtype="float32")
    return {"speech": np.asarray(speech, dtype=np.float32)}, sample_path


def _run_api_check(model: InferenceAPI, sample_path: Path) -> None:
    """Check the contract every front end relies on, on a real bundle."""
    result = model(str(sample_path))
    assert isinstance(result["text"], str), result
    same = model(Audio.read(sample_path, model.sample_rate))
    assert same["text"] == result["text"], (same, result)


def main() -> None:
    args = _parse_args()
    pack_dir = Path(os.environ["PACK_DIR"]).resolve()
    inference_config = pack_dir / "conf" / "inference.yaml"
    meta_path = pack_dir / "meta.yaml"

    if not inference_config.is_file():
        raise FileNotFoundError(
            f"Packed inference config not found: {inference_config}"
        )
    if not meta_path.is_file():
        raise FileNotFoundError(f"Packed metadata not found: {meta_path}")

    _, sample_path = _load_sample_audio()
    _run_api_check(load(pack_dir), sample_path)

    if args.model_tag:
        _run_api_check(load(args.model_tag), sample_path)


if __name__ == "__main__":
    main()
