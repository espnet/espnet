"""Checking a metric's declared inputs/outputs against what it is given."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from espnet3.api.inference import KINDS, Field

logger = logging.getLogger(__name__)

_WARNED_UNDECLARED: set = set()


class MetricContractError(ValueError):
    """A metric's declaration does not match reality.

    Raised when what inference produced does not match a metric's
    declared inputs, or when a metric's return value does not match its
    declared outputs.

    Examples:
        >>> raise MetricContractError("WER.outputs field 'wer' must be number")
        Traceback (most recent call last):
        espnet3.components.contract.metrics.MetricContractError: WER.outputs field ...
    """


def strict_contracts_enabled() -> bool:
    """Whether ``ESPNET3_STRICT_CONTRACTS`` asks for errors instead of warnings.

    Examples:
        >>> import os
        >>> _ = os.environ.pop("ESPNET3_STRICT_CONTRACTS", None)
        >>> strict_contracts_enabled()
        False
        >>> os.environ["ESPNET3_STRICT_CONTRACTS"] = "1"
        >>> strict_contracts_enabled()
        True
        >>> del os.environ["ESPNET3_STRICT_CONTRACTS"]
    """
    return os.environ.get("ESPNET3_STRICT_CONTRACTS", "") not in ("", "0")


def warn_undeclared(cls: type, attr: str) -> None:
    """Warn once per class that it has no ``attr`` contract declaration.

    Raises ``TypeError`` instead, under ``ESPNET3_STRICT_CONTRACTS``.

    Examples:
        >>> import os
        >>> os.environ["ESPNET3_STRICT_CONTRACTS"] = "1"
        >>> class Undeclared:
        ...     pass
        >>> warn_undeclared(Undeclared, "outputs")
        Traceback (most recent call last):
        TypeError: Undeclared does not declare outputs; its inputs/outputs are ...
        >>> del os.environ["ESPNET3_STRICT_CONTRACTS"]
    """
    message = (
        f"{cls.__qualname__} does not declare {attr}; its inputs/outputs are "
        "not checked. Add `inputs`/`outputs` class attributes to opt in."
    )
    if strict_contracts_enabled():
        raise TypeError(message)
    if cls not in _WARNED_UNDECLARED:
        _WARNED_UNDECLARED.add(cls)
        logger.warning(message)


def check_metric_declaration(cls: type) -> None:
    """Raise ``TypeError`` unless every declared output has kind ``number``.

    Called in addition to the shared ``check_declaration`` (non-empty
    tuples, distinct names, ...); this is the one rule specific to metrics.

    Examples:
        >>> class BadMetric:
        ...     outputs = (Field("transcript", "text"),)
        >>> check_metric_declaration(BadMetric)
        Traceback (most recent call last):
        TypeError: BadMetric.outputs field 'transcript' must have kind 'number', ...
    """
    for f in cls.outputs:
        if f.kind != "number":
            raise TypeError(
                f"{cls.__qualname__}.outputs field {f.name!r} must have kind "
                f"'number', not {f.kind!r}"
            )


def read_fields_json(test_dir: Path) -> Optional[Mapping[str, Any]]:
    """Read ``<test_dir>/fields.json``, or ``None`` if it does not exist.

    An inference output directory predating this contract (or written by a
    custom runner that does not write ``fields.json``) has no such file;
    callers fall back to checking SCP presence only, not kind.

    Examples:
        >>> import tempfile
        >>> test_dir = Path(tempfile.mkdtemp())
        >>> read_fields_json(test_dir)
        >>> _ = (test_dir / "fields.json").write_text(
        ...     '{"fields": {"ref": {"kind": "text"}}}')
        >>> read_fields_json(test_dir)
        {'fields': {'ref': {'kind': 'text'}}}
    """
    path = Path(test_dir) / "fields.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _alias_map(metric_config: Any) -> Mapping[str, str]:
    """Map declared input name -> SCP file name, from ``metric_config.inputs``.

    ``metric_config.inputs`` may be a list (alias and file name are the
    same) or a mapping (alias -> file name); absent entirely, every name
    maps to itself.
    """
    inputs = (
        getattr(metric_config, "inputs", None) if metric_config is not None else None
    )
    if inputs is None:
        return {}
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return {name: name for name in inputs}


def _explain(
    metric: Any,
    test_name: str,
    inference_dir: Path,
    written: Optional[Mapping[str, Any]],
    problems: list,
) -> str:
    lines = [
        f"{type(metric).__name__} cannot score test set {test_name!r} from "
        f"{Path(inference_dir) / test_name}:"
    ]
    lines.extend(f"  - {p}" for p in problems)
    if written is not None:
        wrote = ", ".join(
            f"{name} ({info.get('kind')})" for name, info in written["fields"].items()
        )
        lines.append(f"  inference wrote: {wrote}")
    lines.append(
        "  Fix: make the inference output_fn emit the missing field, or map "
        "it in the metrics config:\n"
        "    metrics:\n"
        "      - metric: {...}\n"
        "        inputs: {<declared name>: <scp name>}"
    )
    return "\n".join(lines)


def check_metric_inputs(
    metric: Any,
    metric_config: Any,
    inference_dir: Path,
    test_name: str,
) -> dict:
    """Map ``metric``'s declared inputs to SCP paths, or explain what is missing.

    Args:
        metric: A ``BaseMetric`` instance. If it declares ``inputs`` (via
            ``input_fields()``), each declared :class:`Field` is matched
            against what inference wrote. A metric with no declaration is
            left to its own ``ref_key``/``hyp_key`` handling; this function
            returns ``{}`` for it (callers fall back as today).
        metric_config: The metric's config node, for its optional ``inputs``
            alias mapping (declared name -> SCP file name).
        inference_dir: Base hypothesis/reference directory.
        test_name: Test set name (a subdirectory of ``inference_dir``).

    Returns:
        dict[str, Path]: Declared input name -> resolved SCP path, for every
        declared (and present) input.

    Raises:
        MetricContractError: A required input is missing, or its kind
            disagrees with what inference wrote (per ``fields.json``, when
            present).

    Examples:
        >>> import tempfile
        >>> from espnet3.components.metrics.base_metric import BaseMetric
        >>> class ExampleMetric(BaseMetric):
        ...     inputs = (Field("ref", "text"),)
        ...     outputs = (Field("score", "number"),)
        ...     def __call__(self, data, test_name, output_dir):
        ...         return {"score": 0.0}
        >>> inference_dir = Path(tempfile.mkdtemp())
        >>> test_dir = inference_dir / "test"
        >>> test_dir.mkdir()
        >>> _ = (test_dir / "ref.scp").write_text("utt1 hello\\n")
        >>> paths = check_metric_inputs(ExampleMetric(), None, inference_dir, "test")
        >>> paths["ref"].name
        'ref.scp'
    """
    input_fields = getattr(metric, "input_fields", None)
    if input_fields is None:
        return {}
    fields = input_fields()
    if not fields:
        return {}

    aliases = _alias_map(metric_config)
    test_dir = Path(inference_dir) / test_name
    written = read_fields_json(test_dir)

    data: dict = {}
    problems: list = []
    for f in fields:
        fname = aliases.get(f.name, f.name)
        path = test_dir / f"{fname}.scp"
        if not path.exists():
            if f.optional:
                continue
            problems.append(f"input {f.name!r} ({f.kind}) -> {fname}.scp is missing")
            continue
        if written is not None:
            kind = written["fields"].get(fname, {}).get("kind")
            if kind is not None and kind != f.kind:
                problems.append(
                    f"input {f.name!r} wants {f.kind} but {fname}.scp holds {kind}"
                )
        data[f.name] = path

    if problems:
        raise MetricContractError(
            _explain(metric, test_name, inference_dir, written, problems)
        )
    if written is None:
        logger.warning(
            "%s: no fields.json under %s; input kinds were not checked",
            type(metric).__name__,
            test_dir,
        )
    return data


def check_metric_output(metric: Any, result: Mapping[str, Any]) -> None:
    """Raise unless ``result`` has every declared output, each of kind ``number``.

    A metric with no ``outputs`` declaration is not checked (as today).

    Args:
        metric: A ``BaseMetric`` instance.
        result: What ``metric(...)`` returned.

    Raises:
        MetricContractError: A declared output is missing from ``result``,
            or its value is not the declared kind.

    Examples:
        >>> from espnet3.components.metrics.base_metric import BaseMetric
        >>> class ExampleMetric(BaseMetric):
        ...     inputs = (Field("ref", "text"),)
        ...     outputs = (Field("score", "number"),)
        ...     def __call__(self, data, test_name, output_dir):
        ...         return {"score": 0.0}
        >>> check_metric_output(ExampleMetric(), {"score": 4.3})
        >>> check_metric_output(ExampleMetric(), {})
        Traceback (most recent call last):
        espnet3.components.contract.metrics.MetricContractError: ExampleMetric ...
    """
    outputs = getattr(type(metric), "outputs", None)
    if not outputs:
        return
    missing = [f.name for f in outputs if f.name not in result]
    if missing:
        raise MetricContractError(
            f"{type(metric).__name__} declares outputs {[f.name for f in outputs]} "
            f"but returned {sorted(result)}; missing {missing}"
        )
    for f in outputs:
        try:
            KINDS[f.kind].check(result[f.name], f, model=None, output=True)
        except TypeError as e:
            raise MetricContractError(
                f"{type(metric).__name__}.outputs field {f.name!r} must be "
                f"{f.kind}: {e}"
            ) from e


__all__ = [
    "Field",
    "MetricContractError",
    "check_metric_declaration",
    "check_metric_inputs",
    "check_metric_output",
    "read_fields_json",
    "strict_contracts_enabled",
    "warn_undeclared",
]
