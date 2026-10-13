"""Inference runner with output validation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from hydra.utils import get_class
from omegaconf import ListConfig

from espnet2.torch_utils.device_funcs import is_out_of_memory_error
from espnet3.api.inference import Audio, InferenceAPI
from espnet3.parallel.base_runner import BaseRunner, concatenate_shard_files
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.utils.scp_utils import check_utt_id
from espnet3.utils.writer_utils import write_artifact

logger = logging.getLogger(__name__)

# models already reported as not taking a batch, so that a long test set
# does not repeat the warning once per batch
_WARNED_UNBATCHED: set = set()


def _normalize_key_list(keys) -> List[str]:
    if keys is None:
        return []
    if isinstance(keys, (list, tuple, ListConfig)):
        return list(keys)
    return [keys]


def _input_lengths(inputs_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Length of every array-like input, per key, for an error message."""
    lengths = {}
    for key, values in inputs_dict.items():
        lengths[key] = [
            (v.shape[0] if hasattr(v, "shape") and len(v.shape) > 0 else None)
            for v in values
        ]
    return lengths


def _iter_outputs(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, list):
        outputs: List[Dict[str, Any]] = []
        for item in result:
            outputs.extend(_iter_outputs(item))
        return outputs
    return [result]


def declared_input_names(config) -> Optional[List[str]]:
    """Return the input field names the configured model class declares.

    Lets ``infer()`` tell an :class:`InferenceAPI` subclass from a bare
    model without building it - an ``Inference`` takes no ``input_key`` -
    and name its inputs in the error when one is set anyway. ``None`` when
    ``model._target_`` names anything else or nothing.

    Args:
        config: The inference config; only ``model._target_`` is read.

    Returns:
        The declared input names, optional ones included, or ``None``.

    Examples:
        >>> cfg = OmegaConf.create(
        ...     {"model": {"_target_": "espnet3.systems.esp2_asr.inference.Inference"}}
        ... )
        >>> declared_input_names(cfg)
        ['speech']
    """
    target = getattr(getattr(config, "model", None), "_target_", None)
    if not isinstance(target, str) or not target:
        return None
    try:
        cls = get_class(target)
    except Exception:  # noqa: BLE001 - not an importable class: not ours to judge
        return None
    if isinstance(cls, type) and issubclass(cls, InferenceAPI):
        return [f.name for f in cls.inputs]
    return None


def _declared_fields(model: InferenceAPI, data: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick the declared inputs out of one dataset item."""
    fields = {}
    for f in model.inputs:
        if f.name in data:
            fields[f.name] = data[f.name]
        elif not f.optional:
            raise KeyError(
                f"dataset item has no {f.name!r}, which {type(model).__qualname__} "
                f"needs; it has {sorted(data)}"
            )
    return fields


def _record(
    output: Mapping[str, Any], data: Mapping[str, Any], idx: Any, idx_key: str
) -> Dict[str, Any]:
    """One result as the writers take it: the id first, then the outputs."""
    record: Dict[str, Any] = {idx_key: data.get(idx_key, str(idx))}
    record.update(output)
    return record


def _writable(
    output: Mapping[str, Any], artifact_configs: Mapping[str, Any]
) -> tuple[dict, dict]:
    """Turn contract values into what the writers take; audio brings its rate.

    Returns:
        The record's values, and the artifact configs this record's audio
        needs: WAV at each value's own rate, unless ``output_artifacts``
        says otherwise. They are per record, never written back into the
        shard's shared configs, so one item's rate is not every item's.
    """
    out, own = {}, {}
    for key, value in output.items():
        if isinstance(value, Audio):
            configured = artifact_configs.get(key)
            if configured is None:
                own[key] = {"type": "wav", "sample_rate": value.rate}
            elif configured.get("type", "wav") == "wav" and (
                configured.get("sample_rate") is None
            ):
                own[key] = {**configured, "sample_rate": value.rate}
            # soundfile writes (samples, channels); Audio keeps channels first
            value = value.array.T if value.array.ndim == 2 else value.array
        elif isinstance(value, list) and all(isinstance(v, Mapping) for v in value):
            # segments, as the contract shapes them: the writers take no
            # top-level list, so they become one JSON document - an empty
            # one too, which is what a silent utterance gives
            value = {key: value}
        out[key] = value
    return out, own


def _forward_inference(
    idx,
    dataset,
    model: InferenceAPI,
    *,
    idx_key: str,
    model_kwargs: Mapping[str, Any],
    output_fn: Any,
):
    """Run one item or a batch through an :class:`InferenceAPI` by its declaration.

    The declared inputs are picked out of each item (optional ones when
    present), the model is called through its own entry points -
    ``model(**fields)`` for one item, ``model.batch(items)`` for a batch -
    and the sample id comes from the item (``idx_key``, else the index).
    Only what the model produced is written; a reference for scoring is
    read from the data by ``measure`` (bind ``ref`` to ``dataset:text``
    in the metrics config's ``inputs:``). A configured ``output_fn`` is
    refused rather than ignored: the declaration fixes the outputs.
    """
    if model_kwargs:
        raise TypeError(
            f"an Inference takes no call-time arguments ({sorted(model_kwargs)}); "
            "put them in its model config"
        )
    if output_fn:
        raise TypeError(
            "an Inference writes its declared outputs and applies no output_fn; "
            "drop output_fn (a reference for scoring is read by measure: bind "
            "`ref` to `dataset:<column>` in the metrics config's `inputs:`)"
        )
    batched = isinstance(idx, (list, tuple))
    indices = list(idx) if batched else [idx]
    items = [dataset[i] for i in indices]
    fields = [_declared_fields(model, data) for data in items]
    outputs = model.batch(fields) if batched else [model(**fields[0])]
    records = [
        _record(out, data, i, idx_key)
        for out, data, i in zip(outputs, items, indices, strict=True)
    ]
    return records if batched else records[0]


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
    """Inference runner with strict output-format validation.

    ``forward`` takes one of two paths. For an
    :class:`~espnet3.api.inference.InferenceAPI` it picks the declared
    inputs out of each item, calls ``model(**fields)`` or
    ``model.batch(items)``, and writes each declared output by its kind;
    no ``input_key`` or ``output_fn`` is used. For any other model it
    passes the ``input_key`` fields and shapes the result with the
    recipe's ``output_fn``. The key names are configurable via ``idx_key`` and
    ``hyp_key``/``ref_key``. ``hyp_key`` and ``ref_key`` may be a single
    string or a list of strings to support multiple hypothesis/reference
    fields. ``idx_key`` is the key used to map each inference result to
    its source dataset index when writing SCP files.

    Output format requirements:
        - The result is a dict with the configured keys plus any extra fields.
        - A sample identifier key must exist under ``idx_key`` so SCP outputs
          can map each result back to the corresponding dataset sample.
        - The sample identifier must be a single value, not a list or tuple.
        - ``hyp_key`` and ``ref_key`` values may be scalars or lists/tuples.
          If lists are returned, each entry is written to its own SCP file
          (e.g., ``hyp0.scp``, ``hyp1.scp``).
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
        """Run inference for one or more dataset items and return output dict(s).

        Args:
            idx: Integer index or an iterable of integer indices into the dataset.
            dataset: Dataset providing inference entries.
            model: Inference model callable on the configured input.
            **kwargs: For an ``InferenceAPI``, ``idx_key`` only: the
                declaration gives the inputs and outputs. For any other
                model, ``input_key`` and optionally ``output_fn_path``;
                ``model_kwargs`` passes extra keyword arguments to it.

        Returns:
            Dict containing ``idx`` and output fields for a single item, or a list
            of dicts for batched inputs: the declared outputs for an
            ``InferenceAPI``, else what ``output_fn`` returns.

        Raises:
            RuntimeError: If required input settings are missing.
            KeyError: If required input keys are missing from the dataset item(s).
            RuntimeError: If batched inference fails; includes guidance to disable
                batching when unsupported.

        Notes:
            - ``input_key`` may be a string or a list/tuple of strings.
            - Batched inputs are passed to the model as lists per key; padding is
              the model's responsibility.

        Examples:
            >>> # Single-item inference
            >>> out = InferenceRunner.forward(
            ...     0, dataset=dataset, model=model,
            ...     input_key="speech", output_fn_path="m.mod.out_fn"
            ... )
            >>> # Batched inference
            >>> out = InferenceRunner.forward(
            ...     [0, 1], dataset=dataset, model=model,
            ...     input_key=["speech", "text"], output_fn_path="m.mod.out_fn"
            ... )
        """
        model_kwargs = kwargs.get("model_kwargs") or {}
        if not isinstance(model_kwargs, Mapping):
            raise TypeError("model_kwargs must be a mapping when provided.")
        model_kwargs = dict(model_kwargs)
        if isinstance(model, InferenceAPI):
            # An Inference declares its inputs and fixes its outputs: the
            # declaration drives the run, and no output_fn is applied.
            return _forward_inference(
                idx,
                dataset,
                model,
                idx_key=kwargs.get("idx_key") or "utt_id",
                model_kwargs=model_kwargs,
                output_fn=kwargs.get("output_fn") or kwargs.get("output_fn_path"),
            )
        if "input_key" not in kwargs:
            raise RuntimeError("input_key must be provided for inference.")
        input_key = kwargs["input_key"]
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
            for data in data_batch:
                if key not in data:
                    raise KeyError(f"Input key '{key}' not found in dataset item.")
            inputs_dict[key] = [data[key] for data in data_batch]

        try:
            model_output = model(**inputs_dict, **model_kwargs)
            if output_fn is None:
                return model_output
            return output_fn(data=data_batch, model_output=model_output, idx=indices)
        except Exception as exc:  # noqa: BLE001
            if is_out_of_memory_error(exc):
                # the generic advice below would be wrong here: the model does
                # support batches, the batch was too large
                raise RuntimeError(
                    f"Batched inference ran out of memory on {len(indices)} "
                    f"items (dataset indices {indices}, input lengths "
                    f"{_input_lengths(inputs_dict)}). Lower `batch_size` in the "
                    f"inference config (this batch had {len(indices)} items), "
                    "or sort the test set by length so that long utterances "
                    "are not padded to each other."
                ) from exc
            # Not every model or output_fn takes a list. Before giving up, run
            # the same items one at a time: that keeps `batch_size` safe to set
            # for every model, and only the speed differs.
            try:
                outputs = [
                    InferenceRunner.forward(i, dataset=dataset, model=model, **kwargs)
                    for i in indices
                ]
            except Exception:  # noqa: BLE001
                raise RuntimeError(
                    "Batched inference failed, and so did running the same items "
                    "one at a time; the second traceback is the one to read."
                ) from exc
            name = type(model).__name__
            if name not in _WARNED_UNBATCHED:
                _WARNED_UNBATCHED.add(name)
                logger.warning(
                    f"{name} or the output_fn did not accept a batch of "
                    f"{len(indices)} items ({type(exc).__name__}: {str(exc)[:200]}); "
                    "the items were run one at a time instead. Set `batch_size` "
                    "to null in the inference config to skip the failed attempt, "
                    "or make the model and output_fn accept lists to decode in "
                    "batches."
                )
            return outputs

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
            output, own_configs = _writable(output, writers["artifact_configs"])
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
                    artifact_config=own_configs.get(field_key)
                    or writers["artifact_configs"].get(field_key),
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
    """Import a dotted ``module.function`` path for a non-Inference model.

    output_fn is only for a model that is not an Inference: an Inference
    writes its declared outputs and refuses an output_fn.
    """
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)
