"""Contract layer: checks one part of a system against its declaration.

:mod:`espnet3.api` says what a part declares - its :class:`~espnet3.api.Field`
inputs and outputs - and checks that the declaration itself is well formed
(:func:`~espnet3.api.check_declaration`). This package checks the part
against that declaration at run time: what it is given, what it returns,
and how its config binds it to the other parts.

Checks live here one part at a time: :mod:`.metrics` (a metric's declared
inputs/outputs), :mod:`.data` (a dataset's declared item fields) and
:mod:`.stages` (:class:`~espnet3.contract.stages.StageSpec`). This package
depends only on :mod:`espnet3.api`; it must never import
:mod:`espnet3.components` or :mod:`espnet3.systems` at module level.
"""
