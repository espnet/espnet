"""Inference runner with output validation."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import ListConfig

from espnet2.torch_utils.device_funcs import is_out_of_memory_error
from espnet3.contract.inference import InferenceContract
from espnet3.contract.kinds.audio import Audio
from espnet3.parallel.base_runner import BaseRunner, concatenate_shard_files
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.utils.scp_utils import check_utt_id
from espnet3.utils.writer_utils import write_artifact


def _normalize_key_list(keys) -> List[str]:
    if keys is None:
        return []
    if isinstance(keys, (list, tuple, ListConfig)):
        return list(keys)
    return [keys]


def _input_lengths(inputs_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Length of every array-like input, per key, for an error message."""
    return {
        key: [
            (value.shape[0] if hasattr(value, "shape") and len(value.shape) else None)
            for value in values
        ]
        for key, values in inputs_dict.items()
    }


def _iter_outputs(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, list):
        outputs: List[Dict[str, Any]] = []
        for item in result:
            outputs.extend(_iter_outputs(item))
        return outputs
    return [result]


def _materialize_output_value(
    idx_value,
    field_key: str,
    value,
    output_dir: Path,
    artifact_config: dict | None,
):
    if isinstance(value, (str, int, float, bool)):
        return value
    idx_value = check_utt_id(idx_value)

    if isinstance(value, Audio):
        audio_config = dict(artifact_config or {})
        audio_config.setdefault("type", "wav")
        audio_config.setdefault("sample_rate", value.rate)
        value = (
            value.array.T
            if audio_config["type"] == "wav" and value.array.ndim == 2
            else value.array
        )
        artifact_config = audio_config

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        artifact_dir = output_dir / field_key
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return write_artifact(
            value,
            artifact_dir / str(idx_value),
            field_config=artifact_config,
        ).as_posix()

    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.item()
        artifact_dir = output_dir / field_key
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return write_artifact(
            value,
            artifact_dir / str(idx_value),
            field_config=artifact_config,
        ).as_posix()

    if isinstance(value, (list, tuple)):
        raise TypeError(
            f"Top-level list outputs are not supported for '{field_key}'. "
            "Return a single value per field, or wrap structured content in a "
            "dict so it can be saved as JSON."
        )

    artifact_dir = output_dir / field_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return write_artifact(
        value,
        artifact_dir / str(idx_value),
        field_config=artifact_config,
    ).as_posix()


class InferenceRunner(BaseRunner):
    """Execute a configured backend and persist its formatted results.

    input_key, inference contract and output_artifacts belong to the recipe.
    The runner checks inputs and outputs when a contract is configured.
    Legacy configs may still provide output_fn.
    """

    def __init__(
        self,
        provider: EnvironmentProvider,
        idx_key: str = "utt_id",
        hyp_key: str | Sequence[str] = "hyp",
        ref_key: str | Sequence[str] = "ref",
        **kwargs,
    ) -> None:
        """Initialize the inference runner with output key settings.

        Args:
            provider: Environment provider that supplies dataset/model/env.
            idx_key: Output dict key used as the sample identifier written in
                the first column of each SCP line. This ties each inference
                result back to its dataset sample. Defaults to ``"utt_id"``.
            hyp_key: Hypothesis key or keys expected in the output dict.
            ref_key: Reference key or keys expected in the output dict.
            **kwargs: Forwarded to ``BaseRunner``.
        """
        super().__init__(provider, **kwargs)
        self.idx_key = idx_key
        self.hyp_key = (
            list(hyp_key) if isinstance(hyp_key, (list, tuple, ListConfig)) else hyp_key
        )
        self.ref_key = (
            list(ref_key) if isinstance(ref_key, (list, tuple, ListConfig)) else ref_key
        )

    def resolve_idx_key(self, output: Dict[str, Any]) -> str:
        """Validate that the configured sample-identifier key exists in output."""
        if self.idx_key not in output:
            raise ValueError(
                "Inference output must include the configured sample identifier "
                "key used to map SCP results back to dataset samples. "
                f"idx_key={self.idx_key!r}"
            )
        return self.idx_key

    @staticmethod
    def _validate_output_with_keys(
        output: Dict[str, Any],
        idx_key: str,
        hyp_key,
        ref_key,
    ) -> None:
        if not isinstance(output, dict):
            raise TypeError(
                f"Expected dict output, got {type(output).__name__}: {output}"
            )

        hyp_keys = _normalize_key_list(hyp_key)
        ref_keys = _normalize_key_list(ref_key)
        if idx_key not in output:
            raise ValueError(
                "Inference output must include the configured sample identifier "
                "key used to map SCP results back to dataset samples. "
                f"idx_key={idx_key!r}"
            )
        expected = {idx_key, *hyp_keys, *ref_keys}
        actual = set(output.keys())
        missing = expected - actual
        if missing:
            raise ValueError(
                "Inference output keys must include all required keys. "
                f"missing={sorted(missing)}"
            )

        idx_value = output[idx_key]
        if isinstance(idx_value, (list, tuple)):
            raise TypeError(
                f"'{idx_key}' must be a single value, not {type(idx_value).__name__}"
            )

    @staticmethod
    def _resolve_output_keys(
        output: Dict[str, Any], idx_key: str, output_keys
    ) -> List[str]:
        keys = _normalize_key_list(output_keys)
        if keys:
            return keys
        return [key for key in output.keys() if key != idx_key]

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Call ``model`` with one item or a batch from ``dataset``.

        Example ``conf/inference.yaml``::

            input_key: speech
            model:
              _target_: src.inference.MyBackend
            contract:
              inputs:
                speech: {kind: audio, sample_rate: 16000,
                         rate_policy: resample, label: Input Audio}
              outputs:
                text: {kind: text, normalize: [strip, to_lower],
                       label: Transcription}

        The infer stage and ``InferenceModel`` call
        The shard loop checks contract inputs before calling ``forward`` and
        validates named outputs after it returns::

            dataset[0] = {"utt_id": "u1", "speech": (44100, waveform)}
            # forward sees dataset[0]["speech"] as a 16 kHz waveform array.
            model(speech=waveform_16k)  # -> {"text": "  HELLO  "}
            # boundary returns {"text": "hello", "utt_id": "u1"}

        ``input_key`` must match ``contract.inputs``. Outputs must have the
        declared names; ``{"hyp": ...}`` above raises an error. A bare audio
        array has no rate to check; pass ``Audio``, ``(rate, array)``, or a path
        when the rate matters. ``output_fn`` and ``contract`` cannot coexist.

        To keep a raw ESPnet2 ``Speech2Text`` model, extract its result in a
        custom runner (``batch_size: null``)::

            # inference.yaml:
            # model: {_target_: espnet2.bin.asr_inference.Speech2Text}
            # runner: {_target_: src.inference.ASRRunner}
            class ASRRunner(InferenceRunner):
                @staticmethod
                def forward(idx, dataset=None, model=None, **kwargs):
                    best = model(speech=dataset[idx]["speech"])[0]
                    return {"text": best[0] or ""}

        For a backend that accepts batches, return one mapping per index::

            class BatchTextRunner(InferenceRunner):
                @staticmethod
                def forward(idx, dataset=None, model=None, **kwargs):
                    if isinstance(idx, (list, tuple)):
                        speech = [dataset[i]["speech"] for i in idx]
                        return [{"text": t} for t in model(speech=speech)]
                    return {"text": model(speech=dataset[idx]["speech"])}

        Optional inputs must also appear in ``input_key``::

            input_key: [speech, prompt]
            contract:
              inputs:
                speech: {kind: audio, sample_rate: 16000}
                prompt: {kind: text, optional: true}
              outputs:
                text: {kind: text}
            # Missing prompt -> model(speech=...); present ->
            # model(speech=..., prompt=...). Do not mix in one batch.

        Custom normalizer example::

            # inference.yaml: contract.imports: [src.contracts]
            # contract.outputs.text.normalize: [strip, my_normalizer]
            # src/contracts.py:
            from espnet3.contract import register_normalizer
            register_normalizer("my_normalizer", normalize_text)

        Overrides keep ``@staticmethod`` and ``idx, dataset, model, **kwargs``.
        Contract input is already normalized; return named outputs without
        manually adding ``utt_id``. Direct ``forward`` calls skip contract
        checks. Bundled recipe code needs ``trust_user_code=True`` when loaded.
        Without a contract, the legacy ``output_fn`` path still applies.
        """
        model_kwargs = kwargs.get("model_kwargs") or {}
        if not isinstance(model_kwargs, Mapping):
            raise TypeError("model_kwargs must be a mapping when provided.")
        model_kwargs = dict(model_kwargs)
        if "input_key" not in kwargs:
            raise RuntimeError("input_key must be provided for inference.")
        input_key = kwargs["input_key"]
        contract = kwargs.get("inference_contract")
        optional_keys = (
            {spec.name for spec in contract.inputs if spec.optional}
            if isinstance(contract, InferenceContract)
            else set()
        )
        output_fn = kwargs.get("output_fn")
        if output_fn is None:
            output_fn_path = kwargs.get("output_fn_path")
            output_fn = _load_output_fn(output_fn_path) if output_fn_path else None

        keys = (
            list(input_key)
            if isinstance(input_key, (list, tuple, ListConfig))
            else [input_key]
        )

        is_batched = isinstance(idx, (list, tuple))
        if not is_batched:
            data = dataset[idx]
            inputs_dict = {}
            for key in keys:
                if key not in data:
                    if key in optional_keys:
                        continue
                    raise KeyError(f"Input key '{key}' not found in dataset item.")
                inputs_dict[key] = data[key]
            model_output = model(**inputs_dict, **model_kwargs)
            if output_fn is None:
                return model_output
            return output_fn(data=data, model_output=model_output, idx=idx)

        indices = list(idx)
        data_batch = [dataset[i] for i in indices]
        inputs_dict = {}
        for key in keys:
            present = [key in data for data in data_batch]
            if not all(present):
                if key in optional_keys and not any(present):
                    continue
                raise KeyError(f"Input key '{key}' not found in dataset item.")
            inputs_dict[key] = [data[key] for data in data_batch]

        try:
            model_output = model(**inputs_dict, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_out_of_memory_error(exc):
                raise
            raise RuntimeError(
                f"Batched inference ran out of memory on {len(indices)} "
                f"items (dataset indices {indices}, input lengths "
                f"{_input_lengths(inputs_dict)}). Lower `batch_size` in the "
                f"inference config (this batch had {len(indices)} items), "
                "or sort the test set by length so that long utterances "
                "are not padded to each other."
            ) from exc
        if output_fn is None:
            return model_output
        return output_fn(data=data_batch, model_output=model_output, idx=indices)

    @staticmethod
    def open_writers(
        shard_dir: Optional[Path],
        output_artifacts: Optional[Dict[str, dict]] = None,
        **env,
    ) -> Dict[str, Any]:
        """Open per-shard SCP writers for worker-side inference outputs."""
        return {
            "shard_dir": shard_dir,
            "artifact_configs": output_artifacts or {},
            "scp_handles": {},
            "field_keys": set(),
        }

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        idx_key: str = "utt_id",
        output_keys=None,
        hyp_key=None,
        ref_key=None,
        **env,
    ) -> None:
        """Validate one forward result and stream it into shard-local SCP files."""
        resolved_output_keys = output_keys
        if resolved_output_keys is None:
            resolved_output_keys = [
                *_normalize_key_list(hyp_key),
                *_normalize_key_list(ref_key),
            ]

        shard_dir = writers.get("shard_dir")
        for output in _iter_outputs(result):
            InferenceRunner._validate_output_with_keys(
                output,
                idx_key=idx_key,
                hyp_key=hyp_key,
                ref_key=ref_key,
            )

            field_keys = InferenceRunner._resolve_output_keys(
                output,
                idx_key=idx_key,
                output_keys=resolved_output_keys,
            )
            writers["field_keys"].update(field_keys)

            # the id heads every SCP line and names every artifact file
            idx_value = check_utt_id(output[idx_key])
            for field_key in field_keys:
                value = _materialize_output_value(
                    idx_value=idx_value,
                    field_key=field_key,
                    value=output[field_key],
                    output_dir=shard_dir,
                    artifact_config=writers["artifact_configs"].get(field_key),
                )
                handle = writers["scp_handles"].get(field_key)
                if handle is None:
                    handle = (shard_dir / f"{field_key}.scp").open(
                        "w", encoding="utf-8"
                    )
                    writers["scp_handles"][field_key] = handle
                handle.write(f"{idx_value} {value}\n")

    @staticmethod
    def close_writers(
        writers: Dict[str, Any],
        state: Dict[str, Any],
        **env,
    ) -> Optional[Dict[str, Any]]:
        """Close shard-local SCP files and report which output keys were written."""
        for handle in writers.get("scp_handles", {}).values():
            handle.close()
        shard_dir = writers["shard_dir"]
        field_keys = sorted(writers.get("field_keys", []))
        (shard_dir / "field_keys.txt").write_text(
            "\n".join(field_keys) + ("\n" if field_keys else ""),
            encoding="utf-8",
        )
        return None

    def merge(self, shard_dirs: List[Path]) -> None:
        """Merge per-shard SCP files into the test-set output directory.

        Reads ``field_keys.txt`` from each shard to discover output field
        names, then concatenates each ``<field>.scp`` across shards in shard
        order into ``output_dir / shard_subdir``.

        Args:
            shard_dirs: Completed shard directories in shard-id order.

        Raises:
            RuntimeError: If no output keys are found across all shards.
        """
        field_keys = []
        seen = set()
        for shard_dir in shard_dirs:
            keys_path = shard_dir / "field_keys.txt"
            if not keys_path.exists():
                continue
            for key in keys_path.read_text(encoding="utf-8").splitlines():
                if key not in seen:
                    seen.add(key)
                    field_keys.append(key)
        if not field_keys:
            raise RuntimeError("No output keys found in inference results.")

        ordered_shard_dirs = sorted(
            shard_dirs,
            key=lambda path: int(path.name.split(".", 1)[1]),
        )
        base_dir = (
            self.output_dir / self.shard_subdir
            if self.shard_subdir
            else self.output_dir
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        for field_key in field_keys:
            concatenate_shard_files(
                ordered_shard_dirs,
                f"{field_key}.scp",
                base_dir / f"{field_key}.scp",
            )

    def __call__(self, indices: Iterable[int]) -> bool:
        """Run inference, write SCP outputs, and validate output formats.

        Args:
            indices (Iterable[int]): Dataset indices to run inference on.

        Returns:
            bool: True when all results are written to SCP files on disk.

        Raises:
            RuntimeError: If ``output_dir`` was not set on construction.
            RuntimeError: If no output keys are found after all shards finish.

        Example:
            >>> runner = InferenceRunner(
            ...     provider, output_dir="/exp/decode", idx_key="utt_id"
            ... )
            >>> runner(range(len(test_dataset)))
            True
            >>> # one .scp per output (text.scp for ASR) under /exp/decode
        """
        super().__call__(indices)
        return True


@lru_cache(maxsize=None)
def _load_output_fn(path: str):
    """Import the recipe's dotted output-formatting function."""
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)
