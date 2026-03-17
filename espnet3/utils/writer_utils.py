"""Artifact writer helpers for inference outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from hydra.utils import get_object
from omegaconf import DictConfig, OmegaConf


def write_artifact(
    value: Any,
    output_path: Path,
    field_config: dict | DictConfig | None = None,
) -> Path:
    """Write a single inference artifact to disk and return its path.

    This function implements the artifact serialization rules used by
    ``espnet3.systems.base.inference.infer()`` for non-scalar outputs.

    Default serialization rules:

    1. ``dict`` values are saved as JSON.
       Nested lists inside the dict are preserved through JSON serialization.

       Example:

       .. code-block:: python

           write_artifact(
               {"speaker": "spk1", "segments": [1, 3, 8]},
               Path("infer/test-clean/meta/utt1"),
           )

       Result:

       .. code-block:: text

           infer/test-clean/meta/utt1.json

    2. ``numpy.ndarray`` values are saved as ``.npy`` by default.

       Example:

       .. code-block:: python

           write_artifact(
               np.asarray([[1.0, 2.0]]),
               Path("infer/test-clean/posterior/utt1"),
           )

       Result:

       .. code-block:: text

           infer/test-clean/posterior/utt1.npy

    3. CPU ``torch.Tensor`` values are converted to NumPy and saved as
       ``.npy`` by default.

    4. Unknown Python objects are saved with pickle by default.

       Example:

       .. code-block:: python

           write_artifact(
               some_python_object,
               Path("infer/test-clean/debug_obj/utt1"),
           )

       Result:

       .. code-block:: text

           infer/test-clean/debug_obj/utt1.pkl

    5. WAV output is configured explicitly through ``field_config``.
       The input value is still a normal NumPy array (or CPU tensor); the
       writer configuration decides that it should be serialized as ``.wav``.

       Example:

       .. code-block:: python

           write_artifact(
               waveform_numpy,
               Path("infer/test-clean/audio/utt1"),
               field_config={"type": "wav", "sample_rate": 16000},
           )

       Result:

       .. code-block:: text

           infer/test-clean/audio/utt1.wav

    6. Custom serialization can be configured through ``field_config`` using
       a function writer.

       Example config:

       .. code-block:: python

           field_config = {
               "writer": {
                   "_target_": "mypkg.inference.write_custom_artifact",
                   "some_option": 123,
               }
           }

       Expected writer signature:

       .. code-block:: python

           def write_custom_artifact(value, output_path, some_option=123):
               ...
               return output_path

       The custom writer must be a function. It must return the written file
       path, and that path must already exist when returned.

    Unsupported values:

    - Bare top-level ``list`` / ``tuple`` values are not supported by the
      inference entrypoint. If structured content is needed, wrap it in a
      ``dict`` and let it be saved as JSON.
    - CUDA tensors are rejected. Move them to CPU before writing.

    Args:
        value: Artifact value to serialize.
        output_path: Base output path. The file suffix may be adjusted to match
            the resolved writer.
        field_config: Optional writer configuration.

    Returns:
        Path to the written artifact.
    """
    type_name = None
    options: dict[str, Any] = {}
    writer_cfg = None
    config_dict = _to_plain_dict(field_config)

    if config_dict:
        writer_cfg = config_dict.get("writer")
        type_name = config_dict.get("type", type_name)
        options = {k: v for k, v in config_dict.items() if k not in {"type", "writer"}}

    if writer_cfg is not None:
        return _run_custom_writer(
            value,
            output_path=output_path,
            writer_cfg=writer_cfg,
            options=options,
        )

    type_name = type_name or infer_artifact_type(value)
    return _write_builtin_artifact(
        value,
        output_path=output_path,
        type_name=type_name,
        options=options,
    )


def infer_artifact_type(value: Any) -> str:
    """Infer the default artifact type from a Python value."""
    if isinstance(value, torch.Tensor):
        return "npy"
    if isinstance(value, np.ndarray):
        return "npy"
    if isinstance(value, dict):
        return "json"
    return "pickle"


def _to_plain_dict(field_config: dict | DictConfig | None) -> dict[str, Any]:
    """Convert an optional OmegaConf or dict config into a plain dict."""
    if field_config is None:
        return {}
    if OmegaConf.is_config(field_config):
        return OmegaConf.to_container(field_config, resolve=True)
    return dict(field_config)


def _run_custom_writer(
    value: Any,
    output_path: Path,
    writer_cfg: dict | DictConfig,
    options: dict[str, Any],
) -> Path:
    """Resolve and execute a custom artifact writer function."""
    writer_dict = _to_plain_dict(writer_cfg)
    target = writer_dict.pop("_target_")
    writer = get_object(target)
    result = writer(value=value, output_path=output_path, **writer_dict, **options)
    return _validate_written_path(result)


def _write_builtin_artifact(
    value: Any,
    output_path: Path,
    type_name: str,
    options: dict[str, Any],
) -> Path:
    """Write an artifact using one of the built-in serialization types."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            raise ValueError(
                "CUDA tensors are not supported for artifact writing. "
                "Move the tensor to CPU before returning it from inference."
            )
        value = value.detach().cpu().numpy()

    output_path = output_path.with_suffix(_suffix_for_type(type_name))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if type_name == "wav":
        sample_rate = options.get("sample_rate")
        if sample_rate is None:
            raise ValueError("WAV artifacts require `sample_rate`.")
        sf.write(output_path, np.asarray(value), int(sample_rate))
    elif type_name == "npy":
        np.save(output_path, np.asarray(value))
    elif type_name == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    elif type_name == "pickle":
        with open(output_path, "wb") as f:
            pickle.dump(value, f)
    else:
        raise ValueError(f"Unsupported artifact type: {type_name}")

    return _validate_written_path(output_path)


def _suffix_for_type(type_name: str) -> str:
    """Return the default filename suffix for a built-in artifact type."""
    suffixes = {
        "wav": ".wav",
        "npy": ".npy",
        "json": ".json",
        "pickle": ".pkl",
    }
    if type_name not in suffixes:
        raise ValueError(f"Unsupported artifact type: {type_name}")
    return suffixes[type_name]


def _validate_written_path(path_like: str | Path) -> Path:
    """Validate that a writer returned an existing path-like artifact target."""
    if not isinstance(path_like, (str, Path)):
        raise TypeError("Artifact writers must return a path-like value.")
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Artifact writer returned a missing path: {path}")
    return path
