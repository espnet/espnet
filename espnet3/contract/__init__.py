"""Contract layer: checks one part of a system against its API declaration.

This package depends only on :mod:`espnet3.api`; it must never import
:mod:`espnet3.components` or :mod:`espnet3.systems`, not even under
``TYPE_CHECKING``. The declaration types
(:class:`~espnet3.api.inference.Field`, :class:`~espnet3.api.inference.Kind`)
and the field-declaration check
(:func:`~espnet3.api.inference.check.check_declaration`) live in
:mod:`espnet3.api.inference`; this package holds :mod:`.metrics` (a
metric's declared inputs/outputs) and :mod:`.stages`
(:class:`~espnet3.contract.stages.StageSpec`). The dataset contract
(:mod:`.data`) follows in #6853.
"""
