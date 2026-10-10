"""Checking a metric's declared inputs/outputs against what it is given."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from espnet3.api.inference import KINDS, Field
from espnet3.components.contract.check import check_declaration

#: A metric input source naming a test-set column instead of an inference
#: SCP file (bind a declared name to ``dataset:text`` in ``inputs:``);
#: must match ``espnet3.systems.base.metric.DATASET_PREFIX``.
DATASET_PREFIX = "dataset:"


class MetricContractError(ValueError):
    """A metric's declaration does not match reality.

    Raised when a metric's declared inputs do not match what the
    configured model declares, or when a metric's return value does not
    match its declared outputs.

    Examples:
        >>> raise MetricContractError("WER.outputs field 'wer' must be number")
        Traceback (most recent call last):
        espnet3.components.contract.metrics.MetricContractError: WER.outputs field ...
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
        ...     def __call__(self, data, test_name, output_dir):
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


def declared_outputs(config: Any) -> Optional[tuple]:
    """Return the configured model's declared outputs, or ``None``.

    Mirrors ``espnet3.systems.base.inference_runner.declared_input_names``:
    the model class is found via ``get_class`` without building it, so
    this touches no checkpoint and no device. ``None`` means the
    configured model is not declared at all - unset, unimportable, or not
    an :class:`~espnet3.api.inference.InferenceAPI` subclass.

    ``outputs`` is read from the class, as the contract declares it
    (``ClassVar``, checked at class definition): unlike ``BaseMetric``,
    no ``InferenceAPI`` subclass in ``espnet3`` sets ``self.outputs``.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create(
        ...     {"model": {"_target_": "espnet3.systems.esp2_asr.inference.Inference"}}
        ... )
        >>> [f.name for f in declared_outputs(cfg)]
        ['text']
    """
    from hydra.utils import get_class

    from espnet3.api.inference import InferenceAPI

    target = getattr(getattr(config, "model", None), "_target_", None)
    if not isinstance(target, str) or not target:
        return None
    try:
        cls = get_class(target)
    except Exception:
        return None
    if isinstance(cls, type) and issubclass(cls, InferenceAPI):
        return cls.outputs
    return None


def _declared_inputs(metric: Any, metric_config: Any) -> Mapping[str, str]:
    """Map each declared input's name -> the source ``metric_config.inputs`` gives.

    The one place a metric's inputs are bound to a source: ``inputs:`` is
    keyed by the declared :class:`~espnet3.api.inference.Field` name
    (``WER``'s ``ref``/``hyp``), the same key the metric itself reads
    ``data`` by - there is no separate renaming step. It may be a mapping
    (name -> source) or a list (name and source are the same, e.g.
    ``inputs: [ref, hyp]``). Required for every non-optional declared
    input; an optional one may be left out.

    Raises:
        MetricContractError: ``metric_config`` has no ``inputs:`` at all
            while the metric declares a required input, a required name
            is missing from it, or it names something the metric does
            not declare.
    """
    fields = getattr(metric, "inputs", ())
    declared = {f.name for f in fields}
    required = {f.name for f in fields if not f.optional}

    raw = getattr(metric_config, "inputs", None) if metric_config is not None else None
    if raw is None:
        example = "\n".join(f"  {f.name}: <source>" for f in fields)
        raise MetricContractError(
            f"{type(metric).__name__} declares inputs {sorted(declared)}; bind "
            "them in the metrics config, one source per name:\n"
            f"inputs:\n{example}"
        )
    inputs = dict(raw) if isinstance(raw, Mapping) else {name: name for name in raw}

    missing = required - set(inputs)
    if missing:
        raise MetricContractError(
            f"{type(metric).__name__} declares required inputs {sorted(missing)}, "
            f"missing from the config's inputs: names {sorted(inputs)}"
        )
    extra = set(inputs) - declared
    if extra:
        raise MetricContractError(
            f"{type(metric).__name__} config inputs {sorted(extra)} are not "
            f"declared; declares {sorted(declared)}"
        )
    return inputs


def check_metric_inputs(
    metric: Any, metric_config: Any, inference_config: Any
) -> Optional[Mapping[str, str]]:
    """Raise unless each declared input matches what the configured model declares.

    Checked once per metric (not once per test set: the declaration does
    not vary by test set). A plain source (not ``dataset:<column>``) must
    name one of the model's declared ``outputs``, of the same kind; a
    ``dataset:<column>`` source is the test set's own data rather than
    something inference wrote, so it is not checked here.

    Args:
        metric: A ``BaseMetric`` instance.
        metric_config: The metric's config node, for its ``inputs``
            mapping (declared name -> source); see :func:`_declared_inputs`.
        inference_config: The inference config; only ``model._target_`` is
            read (see :func:`declared_outputs`).

    Returns:
        The same name -> source mapping :func:`_declared_inputs` built,
        for ``measure`` to resolve to files with (no second parse of
        ``metric_config.inputs``); ``None`` if the metric declares no
        inputs at all.

    Raises:
        MetricContractError: ``inputs:`` is missing or does not match the
            declaration (see :func:`_declared_inputs`), the configured
            model declares no outputs at all, a plain-source input names
            none of them, or a kind disagrees.

    Examples:
        >>> from espnet3.components.metrics.base_metric import BaseMetric
        >>> from omegaconf import OmegaConf
        >>> class ExampleMetric(BaseMetric):
        ...     inputs = (Field("ref", "text"),)
        ...     outputs = (Field("score", "number"),)
        ...     def __call__(self, data, test_name, output_dir):
        ...         return {"score": 0.0}
        >>> cfg = OmegaConf.create(
        ...     {"model": {"_target_": "espnet3.systems.esp2_asr.inference.Inference"}}
        ... )
        >>> metric_cfg = OmegaConf.create({"inputs": {"ref": "text"}})
        >>> check_metric_inputs(ExampleMetric(), metric_cfg, cfg)
        {'ref': 'text'}
    """
    fields = getattr(metric, "inputs", None)
    if not fields:
        return None

    inputs = _declared_inputs(metric, metric_config)
    outputs = None
    outputs_checked = False

    for f in fields:
        source = inputs.get(f.name)
        if source is None:
            continue
        if str(source).startswith(DATASET_PREFIX):
            continue
        if not outputs_checked:
            outputs = declared_outputs(inference_config)
            outputs_checked = True
        if outputs is None:
            target = getattr(getattr(inference_config, "model", None), "_target_", None)
            raise MetricContractError(
                f"inference config model {target!r} declares no outputs; "
                "name an espnet3.api.inference.InferenceAPI subclass as model"
            )
        match = next((o for o in outputs if o.name == source), None)
        if match is None:
            raise MetricContractError(
                f"{type(metric).__name__} wants input {f.name!r} -> {source!r}, "
                "but the inference config's model declares outputs "
                f"{[o.name for o in outputs]}; bind {f.name!r} to one of them "
                "in the metrics config's inputs:, or to a `dataset:<column>`"
            )
        if match.kind != f.kind:
            raise MetricContractError(
                f"{type(metric).__name__} input {f.name!r} wants kind {f.kind!r} "
                f"but the inference config's model declares {source!r} as "
                f"{match.kind!r}"
            )
    return inputs


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
        ...     def __call__(self, data, test_name, output_dir):
        ...         return {"score": 0.0}
        >>> check_metric_output(ExampleMetric(), {"score": 4.3})
        >>> check_metric_output(ExampleMetric(), {})
        Traceback (most recent call last):
        espnet3.components.contract.metrics.MetricContractError: ExampleMetric ...
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


__all__ = [
    "DATASET_PREFIX",
    "Field",
    "MetricContractError",
    "check_metric_declaration",
    "check_metric_inputs",
    "check_metric_output",
    "declared_outputs",
    "require_metric_declaration",
]
