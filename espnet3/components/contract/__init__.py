"""Contract-checking code shared by what declares typed inputs and outputs.

The declaration types themselves (:class:`~espnet3.api.inference.Field`,
:class:`~espnet3.api.inference.Kind`) live in :mod:`espnet3.api.inference`;
this package holds only the checking code built on them:
:mod:`.check` for the field-declaration rule :class:`InferenceAPI` and
:class:`~espnet3.components.metrics.base_metric.BaseMetric` share, and
:mod:`.metrics` for a metric's declared inputs/outputs.
"""
