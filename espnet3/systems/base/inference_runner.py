"""Inference runner with output validation."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, Iterable, List, Sequence

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider


class InferenceRunner(BaseRunner):
    """Inference runner with strict output-format validation.

    This runner implements ``forward`` to call a recipe-provided output
    function. The key names are configurable via ``idx_key`` and
    ``hyp_key``/``ref_key``. ``hyp_key`` and ``ref_key`` may be a single
    string or a list of strings to support multiple hypothesis/reference
    fields.

    Output format requirements:
        - The result is a dict with exactly the configured keys.
        - ``idx_key`` value is a scalar (list/tuple is not allowed).
        - ``hyp_key`` and ``ref_key`` values may be scalars or lists/tuples.
          If lists are returned, each entry is written to its own SCP file
          (e.g., ``hyp0.scp``, ``hyp1.scp``).
    """

    def __init__(
        self,
        provider: EnvironmentProvider,
        *,
        idx_key: str = "idx",
        hyp_key: str | Sequence[str] = "hyp",
        ref_key: str | Sequence[str] = "ref",
        **kwargs,
    ) -> None:
        """Initialize the inference runner with output key settings."""
        super().__init__(provider, **kwargs)
        self.idx_key = idx_key
        self.hyp_key = list(hyp_key) if isinstance(hyp_key, (list, tuple)) else hyp_key
        self.ref_key = list(ref_key) if isinstance(ref_key, (list, tuple)) else ref_key

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
        expected = {self.idx_key, *hyp_keys, *ref_keys}
        actual = set(output.keys())
        missing = expected - actual
        if missing:
            raise ValueError(
                "Inference output keys must include all required keys. "
                f"missing={sorted(missing)}"
            )

        idx_value = output[self.idx_key]
        if isinstance(idx_value, (list, tuple)):
            raise TypeError(
                f"'{self.idx_key}' must be a scalar, got {type(idx_value).__name__}"
            )

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """Run inference for one dataset item and return output dict.

        Args:
            idx: Integer index into the dataset.
            dataset: Dataset providing inference entries.
            model: Inference model callable on the configured input.
            **kwargs: Expects ``input_key`` and ``output_fn_path``.

        Returns:
            Dict containing ``uttid`` and any output fields required for SCPs.

        Raises:
            RuntimeError: If required I/O settings are missing.
        """
        data = dataset[idx]
        if "input_key" not in kwargs:
            raise RuntimeError("input_key must be provided for inference.")
        input_key = kwargs["input_key"]
        output_fn_path = kwargs.get("output_fn_path")
        if not output_fn_path:
            raise RuntimeError("output_fn_path must be provided for inference.")
        output_fn = _load_output_fn(output_fn_path)

        if isinstance(input_key, (list, tuple)):
            model_inputs = []
            for key in input_key:
                if key not in data:
                    raise KeyError(f"Input key '{key}' not found in dataset item.")
                model_inputs.append(data[key])
            model_output = model(*model_inputs)
        else:
            if input_key not in data:
                raise KeyError(f"Input key '{input_key}' not found in dataset item.")
            model_output = model(data[input_key])

        return output_fn(data=data, model_output=model_output, idx=idx)

    @classmethod
    def batch_forward(cls, indices, *, dataset=None, model=None, **kwargs):
        """Run inference for a batch of dataset items and return output dicts.

        Args:
            indices: Iterable of integer indices into the dataset.
            dataset: Dataset providing inference entries.
            model: Inference model callable on the configured input.
            **kwargs: Expects ``input_key`` and ``output_fn_path``.

        Returns:
            List of dicts containing ``uttid`` and any output fields required
            for SCPs.

        Raises:
            RuntimeError: If required I/O settings are missing.
        """
        data_batch = [dataset[i] for i in indices]
        if "input_key" not in kwargs:
            raise RuntimeError("input_key must be provided for inference.")
        input_key = kwargs["input_key"]
        output_fn_path = kwargs.get("output_fn_path")
        if not output_fn_path:
            raise RuntimeError("output_fn_path must be provided for inference.")
        output_fn = _load_output_fn(output_fn_path)

        if isinstance(input_key, (list, tuple)):
            inputs_dict = {}
            for key in input_key:
                for data in data_batch:
                    if key not in data:
                        raise KeyError(f"Input key '{key}' not found in dataset item.")
                inputs_dict[key] = [data[key] for data in data_batch]
        else:
            for data in data_batch:
                if input_key not in data:
                    raise KeyError(f"Input key '{input_key}' not found in dataset item.")
            inputs_dict = {input_key: [data[input_key] for data in data_batch]}

        if hasattr(model, "batch_forward") and callable(model.batch_forward):
            model_output = model.batch_forward(**inputs_dict)
            return output_fn(
                data=data_batch, model_output=model_output, idx=list(indices)
            )

        outputs = []
        for i, data in zip(indices, data_batch):
            if isinstance(input_key, (list, tuple)):
                model_output = model(*[data[key] for key in input_key])
            else:
                model_output = model(data[input_key])
            outputs.append(output_fn(data=data, model_output=model_output, idx=i))

        return outputs

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
