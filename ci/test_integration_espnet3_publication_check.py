#!/usr/bin/env python3

"""Validate a packed ESPnet3 publication bundle with InferenceModel."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from espnet3.components.data.dataset_module import instantiate_dataset_reference
from espnet3.publication import InferenceModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Optional remote model tag checked via InferenceModel.from_pretrained().",
    )
    return parser.parse_args()


def _load_dataset_sample(
    inference_config_path: Path,
    split: str,
    recipe_dir: Path,
) -> dict[str, Any]:
    inference_config = OmegaConf.load(inference_config_path)
    dataset_config = getattr(inference_config, "dataset", None)
    test_entries = getattr(dataset_config, "test", None) if dataset_config else None
    if not test_entries:
        raise ValueError(
            "No dataset.test entries found in inference config: "
            f"{inference_config_path}"
        )

    matched_entry = None
    for entry in test_entries:
        entry_name = getattr(entry, "name", None)
        data_src_args = getattr(entry, "data_src_args", None)
        entry_split = getattr(data_src_args, "split", None) if data_src_args else None
        if entry_name == split or entry_split == split:
            matched_entry = entry
            break

    if matched_entry is None:
        raise ValueError(
            f"No dataset.test entry matched split {split!r} in {inference_config_path}"
        )

    dataset = instantiate_dataset_reference(matched_entry, recipe_dir=recipe_dir)
    sample = dataset[0]
    if not isinstance(sample, dict):
        raise TypeError(
            f"Expected dict sample from dataset entry {split!r}, got {type(sample)!r}"
        )
    return sample


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

    sample = _load_dataset_sample(
        inference_config_path=inference_config,
        split=args.split,
        recipe_dir=Path(args.recipe_dir).resolve(),
    )
    _run_smoke_check(InferenceModel.from_packed(pack_dir, trust_user_code=True), sample)

    if args.model_tag:
        _run_smoke_check(
            InferenceModel.from_pretrained(args.model_tag, trust_user_code=True),
            sample,
        )


if __name__ == "__main__":
    main()
