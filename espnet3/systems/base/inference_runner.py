"""Inference runner with output validation."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omegaconf import ListConfig

from espnet3.parallel.base_runner import BaseRunner, cat_shard_files
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.utils.writer_utils import write_artifact


def _normalize_key_list(keys) -> List[str]:
    if keys is None:
        return []
    if isinstance(keys, (list, tuple, ListConfig)):
        return list(keys)
    return [keys]


def _iter_outputs(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, list):
        outputs: List[Dict[str, Any]] = []
        for item in result:
            outputs.extend(_iter_outputs(item))
        return outputs
    return [result]


def _is_scp_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def _materialize_output_value(
    *,
    idx_value,
    field_key: str,
    value,
    output_dir: Path,
    artifact_config: dict | None,
):
    if _is_scp_scalar(value):
        return value

    if isinstance(value, (list, tuple)):
        raise TypeError(
            f"Top-level list outputs are not supported for '{field_key}'. "
            "Return a single value per field, or wrap structured content in a "
            "dict so it can be saved as JSON."
        )

    artifact_dir = output_dir / field_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / str(idx_value)
    artifact_path = write_artifact(value, target, field_config=artifact_config)
    return artifact_path.as_posix()


class InferenceRunner(BaseRunner):
    """Inference runner with strict output-format validation.

    This runner implements ``forward`` to call a recipe-provided output
    function. The key names are configurable via ``idx_key`` and
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

    def _validate_output(self, output: Dict[str, Any]) -> None:
        self._validate_output_with_keys(
            output,
            idx_key=self.idx_key,
            hyp_key=self.hyp_key,
            ref_key=self.ref_key,
        )

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Run inference for one or more dataset items and return output dict(s).

        Args:
            idx: Integer index or an iterable of integer indices into the dataset.
            dataset: Dataset providing inference entries.
            model: Inference model callable on the configured input.
            **kwargs: Expects ``input_key`` and optionally ``output_fn_path``.

        Returns:
            Dict containing ``idx`` and output fields for a single item, or a list
            of dicts for batched inputs (as returned by ``output_fn``).

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
        if "input_key" not in kwargs:
            raise RuntimeError("input_key must be provided for inference.")
        input_key = kwargs["input_key"]
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
            model_output = model(**inputs_dict)
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
            model_output = model(**inputs_dict)
            if output_fn is None:
                return model_output
            return output_fn(data=data_batch, model_output=model_output, idx=indices)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Batched inference failed. If your model/output_fn does not "
                "support batched inputs, set batch_size to None. "
            ) from exc

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

            if shard_dir is None:
                state.setdefault("records", []).append(output)
                continue

            field_keys = InferenceRunner._resolve_output_keys(
                output,
                idx_key=idx_key,
                output_keys=resolved_output_keys,
            )
            writers["field_keys"].update(field_keys)

            idx_value = output[idx_key]
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
    def close_writers(writers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Close shard-local SCP files and report which output keys were written."""
        for handle in writers.get("scp_handles", {}).values():
            handle.close()
        return {"field_keys": sorted(writers.get("field_keys", []))}

    def merge_shard_files(
        self,
        shard_dirs: List[Path],
        states: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge per-shard SCP files into the test-set output directory."""
        if self.output_dir is None:
            return None

        field_keys = []
        seen = set()
        for state in states:
            for key in state.get("field_keys", []):
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
            cat_shard_files(
                ordered_shard_dirs,
                f"{field_key}.scp",
                base_dir / f"{field_key}.scp",
            )
        return {}

    def __call__(self, indices: Iterable[int]) -> Any:
        """Run inference and validate output formats."""
        results = super().__call__(indices)
        if results is None:
            return None
        if not isinstance(results, list):
            return results

        flat_results: List[Any] = []
        for item in results:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)

        for item in flat_results:
            self._validate_output(item)

        return flat_results


@lru_cache(maxsize=None)
def _load_output_fn(path: str):
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)
