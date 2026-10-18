#!/usr/bin/env python3

"""Validate a packed ESPnet3 publication bundle with InferenceModel."""

from __future__ import annotations

import argparse
import os
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
    asset_name = "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    sample_path = tmp_dir / Path(asset_name).name
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    import torchaudio.utils

    # torchaudio 2.9 removed the public ``download_asset`` helper. The project
    # currently supports torchaudio 2.11, where the same downloader remains
    # available under its internal name.
    download_asset = getattr(torchaudio.utils, "download_asset", None)
    if download_asset is None:
        download_asset = getattr(torchaudio.utils, "_download_asset", None)
    if download_asset is None:
        raise RuntimeError("torchaudio does not provide an asset downloader")

    return Path(download_asset(asset_name, path=sample_path))


def _load_sample_audio() -> dict[str, Any]:
    """Return a minimal ASR-friendly sample dict."""
    temp_root = Path(
        os.environ.get("TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("TMP")
        or "/tmp"
    )
    sample_path = _download_sample_audio(temp_root / "espnet3-publication-assets")
    speech, sample_rate = sf.read(str(sample_path), dtype="float32")
    return {"speech": (sample_rate, np.asarray(speech, dtype=np.float32))}


def _validate_output(result: Any, expected_utt_id: str | None = None) -> None:
    assert isinstance(result, dict)
    assert set(result) == {"utt_id", "text"}
    assert isinstance(result["text"], str)
    if expected_utt_id is not None:
        assert result["utt_id"] == expected_utt_id


def _run_smoke_check(session: InferenceModel, sample: dict[str, Any]) -> None:
    assert session.contract is not None
    assert [spec.name for spec in session.contract.inputs] == ["speech"]
    assert [spec.name for spec in session.contract.outputs] == ["text"]
    assert session.contract.inputs[0].options["sample_rate"] == 16000
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
    """Check the packed model and, optionally, its remote publication."""
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
