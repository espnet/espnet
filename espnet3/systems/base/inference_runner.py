from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Iterable, List, Sequence

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider


class AbsInferenceRunner(BaseRunner, ABC):
    """Base runner with strict output-format validation.

    Subclasses must implement ``forward`` to return a dictionary containing
    keys that identify index, hypothesis, and reference. The key names are
    configurable via ``idx_key`` and ``hyp_key``/``ref_key``. ``hyp_key`` and
    ``ref_key`` may be a single string or a list of strings to support multiple
    hypothesis/reference fields.

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
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(
                "Inference output keys must match expected keys. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        idx_value = output[self.idx_key]
        if isinstance(idx_value, (list, tuple)):
            raise TypeError(
                f"'{self.idx_key}' must be a scalar, got {type(idx_value).__name__}"
            )

    def __call__(self, indices: Iterable[int]) -> List[Any] | None:
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
