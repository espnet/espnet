"""Checking a metric's declared inputs/outputs against what it is given."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from espnet3.contract import KINDS, Field, check_declaration

#: A metric input source naming a test-set column instead of an inference
#: SCP file (bind a declared name to ``dataset:text`` in ``inputs:``);
#: must match ``espnet3.systems.base.metric.DATASET_PREFIX``.
DATASET_PREFIX = "dataset:"


class MetricContractError(ValueError):
    """A metric's declaration does not match reality.

    Raised when metric inputs or outputs do not match their declarations.

    Examples:
        >>> raise MetricContractError("WER.outputs field 'wer' must be number")
        Traceback (most recent call last):
        espnet3.contract.metrics.MetricContractError: WER.outputs field ...
    """


def require_metric_declaration(obj) -> None:
    """Raise unless ``obj`` declares ``inputs``/``outputs``.

    Every ``BaseMetric`` instance must opt into the contract - as its
    class's own attributes, or as the instance's own (set in
    ``__init__``, for a metric whose contract depends on its own
    configuration); there is no undeclared fallback.

    Args:
        obj: The class to check, or an instance whose own attributes (set
            in its ``__init__``) declare a contract its class does not.

    Examples:
        >>> class Undeclared:
        ...     pass
        >>> require_metric_declaration(Undeclared)
        Traceback (most recent call last):
        TypeError: Undeclared does not declare inputs/outputs; declare ...
    """
    cls = obj if isinstance(obj, type) else type(obj)
    raise TypeError(
        f"{cls.__qualname__} does not declare inputs/outputs; declare "
        "`inputs`/`outputs` (class attributes, or set on `self` before "
        "calling super().__init__())."
    )


def check_metric_declaration(obj) -> None:
    """Raise ``TypeError`` unless every declared output has kind ``number``.

    Called in addition to the shared ``check_declaration`` (non-empty
    tuples, distinct names, ...); this is the one rule specific to metrics.

    Args:
        obj: The class to check, or an instance whose own attributes (set
            in its ``__init__``) declare a contract its class does not.

    Examples:
        >>> class BadMetric:
        ...     outputs = (Field("transcript", "text"),)
        >>> check_metric_declaration(BadMetric)
        Traceback (most recent call last):
        TypeError: BadMetric.outputs field 'transcript' must have kind 'number', ...
    """
    cls = obj if isinstance(obj, type) else type(obj)
    for f in obj.outputs:
        if f.kind != "number":
            raise TypeError(
                f"{cls.__qualname__}.outputs field {f.name!r} must have kind "
                f"'number', not {f.kind!r}"
            )


def check_metric_contract(metric: Any) -> None:
    """Raise unless ``metric`` declares a valid ``inputs``/``outputs`` contract.

    ``BaseMetric.__init__`` already calls this on construction, but a
    subclass that overrides ``__init__`` without calling
    ``super().__init__()`` skips that - so ``measure()`` calls this again
    on every metric it instantiates, before trusting its declaration for
    anything. There is no undeclared fallback either way.

    Args:
        metric: A ``BaseMetric`` instance (or any object exposing
            ``inputs``/``outputs`` the same way).

    Raises:
        TypeError: Neither ``inputs`` nor ``outputs`` is declared, or the
            declaration is malformed (see ``check_declaration``,
            ``check_metric_declaration``).

    Examples:
        >>> from espnet3.components.metrics.base_metric import BaseMetric
        >>> class Undeclared(BaseMetric):
        ...     def __init__(self):
        ...         pass  # declares nothing, and skips super().__init__()
        ...     def update(self, **values):
        ...         pass
        ...     def compute(self):
        ...         return {}
        >>> metric = Undeclared()  # construction alone does not catch it
        >>> check_metric_contract(metric)
        Traceback (most recent call last):
        TypeError: Undeclared does not declare inputs/outputs; declare ...
    """
    if not (hasattr(metric, "inputs") or hasattr(metric, "outputs")):
        require_metric_declaration(metric)
    check_declaration(metric)
    check_metric_declaration(metric)


def check_metric_inputs(metric: Any, inputs: Any) -> Optional[Mapping[str, str]]:
    """Validate metric input bindings without inspecting an inference model.

    ``inputs`` maps each name declared by the metric to a source: either an
    inference artifact stem or ``dataset:<column>``. Whether an inference
    artifact exists is checked when the source is resolved for a test set.
    """
    fields = getattr(metric, "inputs", ())
    if not fields:
        return None

    if isinstance(inputs, Mapping):
        bindings = dict(inputs)
    else:
        bindings = {name: name for name in inputs}

    if isinstance(fields, tuple) and all(isinstance(field, Field) for field in fields):
        declared = {field.name for field in fields}
        required = {field.name for field in fields if not field.optional}
    else:
        declared = set(fields)
        required = {
            name
            for name, expected in fields.items()
            if not isinstance(None, expected)
        }

    missing = required - set(bindings)
    unknown = set(bindings) - declared
    if missing or unknown:
        raise MetricContractError(
            f"{type(metric).__name__} inputs: missing {sorted(missing)}, "
            f"unknown {sorted(unknown)}"
        )
    for name, source in bindings.items():
        if not isinstance(source, str) or not source or source == DATASET_PREFIX:
            raise MetricContractError(
                f"input {name!r} needs an SCP stem or dataset column"
            )
    return bindings


def check_metric_output(metric: Any, result: Mapping[str, Any]) -> None:
    """Raise unless ``result`` has every declared output, each of kind ``number``.

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
        ...     def update(self, **values):
        ...         pass
        ...     def compute(self):
        ...         return {"score": 0.0}
        >>> check_metric_output(ExampleMetric(), {"score": 4.3})
        >>> check_metric_output(ExampleMetric(), {})
        Traceback (most recent call last):
        espnet3.contract.metrics.MetricContractError: ExampleMetric ...
    """
    outputs = getattr(metric, "outputs", None)
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


def check_metric_batch(metric: Any, values: Mapping[str, Any]) -> None:
    """Raise unless ``values`` is one good batch for ``metric.update``.

    ``values`` maps each declared input name to a list or tuple with one
    element per utterance. Checks, in order: no undeclared name; every
    required name present (an optional one may be absent or ``None``);
    each value a list or tuple; all lengths equal; every element
    accepted by its field's kind (``Kind.accepts``, no model needed).

    Args:
        metric: A ``BaseMetric`` instance.
        values: The keyword arguments given to ``update``.

    Raises:
        MetricContractError: The batch breaks one of the checks. The
            message names the input and, for a bad element, its position
            and the expected kind.

    Examples:
        >>> from espnet3.systems.esp2_asr.metrics.wer import WER
        >>> check_metric_batch(WER(), {"ref": ["a b"], "hyp": ["a c"]})
        >>> check_metric_batch(WER(), {"ref": ["a b"], "hyp": [3]})
        Traceback (most recent call last):
        espnet3.contract.metrics.MetricContractError: WER input ...
    """
    name = type(metric).__name__
    fields = {f.name: f for f in getattr(metric, "inputs", ())}
    declared = ", ".join(fields)
    for key in values:
        if key not in fields:
            raise MetricContractError(
                f"{name} got input {key!r}; declared inputs: {declared}"
            )
    given = {}
    for key, field in fields.items():
        value = values.get(key)
        if value is None:
            if field.optional:
                continue
            raise MetricContractError(
                f"{name} input {key!r} is missing; declared inputs: {declared}"
            )
        if not isinstance(value, (list, tuple)):
            raise MetricContractError(
                f"{name} input {key!r} must be a list or tuple, "
                f"not {type(value).__name__}"
            )
        given[key] = value
    if given:
        first, *rest = given
        for key in rest:
            if len(given[key]) != len(given[first]):
                raise MetricContractError(
                    f"{name} input {key!r} has {len(given[key])} items, "
                    f"{first!r} has {len(given[first])}"
                )
    for key, value in given.items():
        field = fields[key]
        for i, elem in enumerate(value):
            if not KINDS[field.kind].accepts(elem, field):
                raise MetricContractError(
                    f"{name} input {key!r}[{i}] is {type(elem).__name__}, "
                    f"must be {field.kind}"
                )


__all__ = [
    "DATASET_PREFIX",
    "Field",
    "MetricContractError",
    "check_metric_batch",
    "check_metric_declaration",
    "check_metric_inputs",
    "check_metric_output",
    "require_metric_declaration",
]
