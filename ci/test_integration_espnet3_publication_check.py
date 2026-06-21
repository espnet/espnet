#!/usr/bin/env python3

"""Validate a packed ESPnet3 publication bundle with InferenceModel."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from espnet3.publication import InferenceModel


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
        help="Optional remote model tag checked via InferenceModel.from_pretrained().",
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


def _load_sample_audio() -> dict[str, Any]:
    """Return a minimal ASR-friendly sample dict."""
    temp_root = Path(
        os.environ.get("TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("TMP")
        or "/tmp"
    )
    sample_path = _download_sample_audio(temp_root / "espnet3-publication-assets")
    speech, _sample_rate = sf.read(str(sample_path), dtype="float32")
    return {"speech": np.asarray(speech, dtype=np.float32)}


def _is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray)):
        return bool(value)
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return all(int(dim) > 0 for dim in shape)
        except TypeError:
            return True
    try:
        return len(value) > 0
    except TypeError:
        return True


def _validate_output(result: Any, expected_utt_id: str | None = None) -> None:
    if isinstance(result, dict):
        if expected_utt_id is not None:
            assert result.get("utt_id") == expected_utt_id
        if "wav" in result:
            assert _is_nonempty_value(result["wav"])
            return
        if "hyp" in result:
            assert _is_nonempty_value(result["hyp"])
            return
        payload_keys = [key for key in result if key not in {"utt_id", "ref"}]
        assert payload_keys, "Expected at least one payload key in inference output."
        assert any(_is_nonempty_value(result[key]) for key in payload_keys)
        return
    assert _is_nonempty_value(result)


def _run_smoke_check(session: InferenceModel, sample: dict[str, Any]) -> None:
    input_keys = (
        session.input_key
        if isinstance(session.input_key, list)
        else [session.input_key]
    )
    for key in input_keys:
        if key not in sample:
            raise KeyError(
                "Dataset sample is missing required input key "
                f"{key!r}: {sorted(sample)}"
            )

    result = session(sample, idx="publication-test")
    _validate_output(result, expected_utt_id="publication-test")

    batch_result = session.forward_batch([sample])
    assert len(batch_result) == 1
    _validate_output(batch_result[0])


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

    sample = _load_sample_audio()
    _run_smoke_check(InferenceModel.from_packed(pack_dir, trust_user_code=True), sample)

    if args.model_tag:
        _run_smoke_check(
            InferenceModel.from_pretrained(args.model_tag, trust_user_code=True),
            sample,
        )


if __name__ == "__main__":
    main()
