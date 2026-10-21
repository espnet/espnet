#!/usr/bin/env python3

"""Validate a packed ESPnet3 publication bundle with InferenceModel."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from espnet3.publication import InferenceModel
from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS
from espnet3.publication.demo.session import DemoSession
from espnet3.publication.inference_model import _load_inference_config
from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_provider import InferenceProvider


class VerificationProvider(InferenceProvider):
    """Decode the exact oracle WAVs through the real recipe runner."""

    @staticmethod
    def build_dataset(config):
        return [
            {"utt_id": f"oracle-{index}", "speech": path}
            for index, path in enumerate(config.verification_audio_paths)
        ]


def _check_recipe(pack_dir: Path, sample: dict[str, Any], expected: str) -> None:
    with tempfile.TemporaryDirectory(prefix="espnet3-parity-") as temporary:
        root = Path(temporary)
        rate, wave = sample["speech"]
        paths = [root / "full.wav", root / "short.wav", root / "repeat.wav"]
        for path, samples in zip(paths, (wave, wave[: len(wave) // 2], wave)):
            sf.write(path, samples, rate, subtype="FLOAT")
        session = InferenceModel.from_packed(pack_dir, trust_user_code=True)
        oracle = [
            session.model(speech=sf.read(path, dtype="float32")[0])[0][0] or ""
            for path in paths
        ]
        assert oracle[0] == expected
        config = _load_inference_config(pack_dir / "conf" / "inference.yaml", pack_dir)
        config.provider = {"_target_": f"{__name__}.VerificationProvider"}
        config.dataset = {"test": [{"name": "oracle"}]}
        config.verification_audio_paths = [str(path) for path in paths]
        config.parallel = {
            "env": "local",
            "n_workers": 1,
            "options": {"processes": False, "dashboard_address": None},
        }
        for batch_size in (None, 2):
            config.batch_size = batch_size
            config.inference_dir = str(root / f"decode-{batch_size}")
            infer(config)
            lines = (
                (Path(config.inference_dir) / "oracle" / "text.scp")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            assert lines == [
                f"oracle-{index} {text}" for index, text in enumerate(oracle)
            ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-file", type=Path, default=None)
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


def _run_smoke_check(session: InferenceModel, sample: dict[str, Any]) -> str:
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
    rate, wave = sample["speech"]
    samples = [sample, {"speech": (rate, wave[: len(wave) // 2])}, sample]
    # This is the native Speech2Text result, independently of runner formatting.
    expected = [session.model(speech=item["speech"][1])[0][0] or "" for item in samples]
    assert expected[0].strip(), "The speech fixture must produce nonempty text"
    assert result["text"] == expected[0]
    demo = DemoSession(
        Path("."),
        OmegaConf.create(
            {"model": {"call_args": {}}, "ui": {"title": None, "description": None}}
        ),
        session,
        DEFAULT_UI_ASSETS.clone(),
    )
    callback = demo.create_inference_fn(demo.input_specs, demo.output_specs)
    for order in ([0], [0, 1], [1, 2, 0]):
        ordered = [samples[index] for index in order]
        batch_result = session.forward_batch(
            ordered, indices=[f"oracle-{index}" for index in order]
        )
        assert len(batch_result) == len(order)
        for item, index, batch_item in zip(ordered, order, batch_result):
            _validate_output(batch_item, expected_utt_id=f"oracle-{index}")
            assert batch_item["text"] == session(item)["text"] == expected[index]
            assert callback(item["speech"]) == expected[index]
    assert session.forward_batch([]) == []
    # Validation failures must leave the loaded session usable.
    for invalid in (None, (rate, np.array([np.nan], dtype=np.float32))):
        try:
            session({"speech": invalid})
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid audio unexpectedly reached inference")
        assert session(sample)["text"] == expected[0]
    return expected[0]


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
    expected = _run_smoke_check(
        InferenceModel.from_packed(pack_dir, trust_user_code=True), sample
    )
    _check_recipe(pack_dir, sample, expected)
    if args.reference_file is not None:
        # FLOAT preserves exactly the waveform used by the native oracle.
        audio_path = args.reference_file.with_suffix(".wav").resolve()
        sf.write(audio_path, sample["speech"][1], sample["speech"][0], subtype="FLOAT")
        args.reference_file.write_text(
            json.dumps({"audio_path": str(audio_path), "text": expected}),
            encoding="utf-8",
        )

    if args.model_tag:
        _run_smoke_check(
            InferenceModel.from_pretrained(args.model_tag, trust_user_code=True),
            sample,
        )


if __name__ == "__main__":
    main()
