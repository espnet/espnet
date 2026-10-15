"""Metric contracts independent of inference models and their outputs."""

from collections.abc import Mapping
from numbers import Real

from espnet3.contract.data import DatasetContractError, check_fields

DATASET_PREFIX = "dataset:"


class MetricContractError(ValueError):
    """A metric declaration, batch, binding or result is invalid."""


def check_metric_contract(metric):
    """Require input Python types and a nonempty tuple of numeric output names."""
    try:
        check_fields(metric, "inputs")
    except DatasetContractError as error:
        raise MetricContractError(str(error)) from error
    outputs = getattr(metric, "outputs", None)
    if (
        not isinstance(outputs, tuple)
        or not outputs
        or not all(isinstance(name, str) and name for name in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        raise MetricContractError("outputs must be a nonempty tuple of distinct names")


def check_metric_inputs(metric, inputs):
    """Validate explicit input bindings without importing an inference model."""
    declared = metric.inputs
    unknown = set(inputs) - set(declared)
    missing = [
        name
        for name, expected in declared.items()
        if name not in inputs and not isinstance(None, expected)
    ]
    if unknown or missing:
        raise MetricContractError(
            f"{type(metric).__name__} inputs: missing {missing}, "
            f"unknown {sorted(unknown)}"
        )
    for name, source in inputs.items():
        if not isinstance(source, str) or not source or source == DATASET_PREFIX:
            raise MetricContractError(
                f"input {name!r} needs an SCP stem or dataset column"
            )
    return inputs


def check_metric_batch(metric, values):
    """Check required inputs, aligned list/tuple lengths and per-item types."""
    check_metric_inputs(metric, {name: name for name in values})
    lengths = {}
    for name, value in values.items():
        expected = metric.inputs[name]
        if value is None and isinstance(None, expected):
            continue
        if not isinstance(value, (list, tuple)):
            raise MetricContractError(f"input {name!r} must be a list or tuple")
        lengths[name] = len(value)
        for index, item in enumerate(value):
            if not isinstance(item, expected):
                raise MetricContractError(
                    f"input {name!r}[{index}] must match {expected}, "
                    f"got {type(item).__name__}"
                )
    if len(set(lengths.values())) > 1:
        raise MetricContractError(f"metric batch lengths differ: {lengths}")


def check_metric_output(metric, result):
    """Require all declared output names and real numeric values."""
    if not isinstance(result, Mapping):
        raise MetricContractError("metric result must be a mapping")
    missing = set(metric.outputs) - set(result)
    if missing:
        raise MetricContractError(f"metric result is missing {sorted(missing)}")
    for name, value in result.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise MetricContractError(f"metric output {name!r} must be a real number")
