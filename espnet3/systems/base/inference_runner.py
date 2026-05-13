"""Inference runner with output validation."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, Iterable, List, Sequence

from omegaconf import ListConfig

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider


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

    def _validate_output(self, output: Dict[str, Any]) -> None:
        if not isinstance(output, dict):
            raise TypeError(
                f"Expected dict output, got {type(output).__name__}: {output}"
            )

        hyp_keys = (
            list(self.hyp_key)
            if isinstance(self.hyp_key, (list, tuple))
            else [self.hyp_key]
        )
        ref_keys = (
            list(self.ref_key)
            if isinstance(self.ref_key, (list, tuple))
            else [self.ref_key]
        )
        idx_key = self.resolve_idx_key(output)
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

    def __call__(self, indices: Iterable[int]) -> List[Any] | None:
        """Run inference and validate output formats."""
        results = super().__call__(indices)
        if self.async_mode:
            return results
        if results is None:
            return None

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
