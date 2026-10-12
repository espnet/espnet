"""Contract-checking code shared by what declares typed inputs and outputs.

The declaration types (:class:`~espnet3.api.inference.Field`,
:class:`~espnet3.api.inference.Kind`) and the field-declaration check
(:func:`~espnet3.api.inference.check.check_declaration`) live in
:mod:`espnet3.api.inference`; this package holds :mod:`.metrics` (a
metric's declared inputs/outputs) and :mod:`.stages`
(:class:`~espnet3.components.contract.stages.StageSpec`).
"""
