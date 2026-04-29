#!/usr/bin/env python3

"""Validate a packed ESPnet3 publication bundle with InferenceSession."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Sequence

from espnet3.publication import InferenceSession


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-module",
        required=True,
        help="Dataset module exposing `Dataset`, e.g. egs3.mini_an4.asr.dataset.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split used for the publication smoke check.",
    )
    parser.add_argument(
        "--recipe-dir",
        default=".",
        help="Recipe root passed to the dataset constructor.",
    )
    return parser.parse_args()


def _load_dataset_sample(
    *,
    dataset_module: str,
    split: str,
    recipe_dir: Path,
) -> dict[str, Any]:
    dataset_lib = importlib.import_module(dataset_module)
    dataset = dataset_lib.Dataset(split=split, recipe_dir=recipe_dir)
    sample = dataset[0]
    if not isinstance(sample, dict):
        raise TypeError(
            f"Expected dict sample from {dataset_module}.Dataset, got {type(sample)!r}"
        )
    return sample


def _required_keys(input_key: str | Sequence[str]) -> list[str]:
    if isinstance(input_key, str):
        return [input_key]
    return list(input_key)


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


def _validate_output(result: Any, *, expected_utt_id: str | None = None) -> None:
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

    session = InferenceSession.from_artifacts(
        {"inference_config": str(inference_config)},
        trust_user_code=True,
    )

    sample = _load_dataset_sample(
        dataset_module=args.dataset_module,
        split=args.split,
        recipe_dir=Path(args.recipe_dir).resolve(),
    )
    for key in _required_keys(session.input_key):
        if key not in sample:
            raise KeyError(
                f"Dataset sample is missing required input key {key!r}: {sorted(sample)}"
            )

    result = session(sample, idx="publication-test")
    _validate_output(result, expected_utt_id="publication-test")

    batch_result = session.forward_batch([sample])
    assert len(batch_result) == 1
    _validate_output(batch_result[0])


if __name__ == "__main__":
    main()
