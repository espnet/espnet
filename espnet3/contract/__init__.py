"""Contract layer: checks one part of a system against its API declaration.

Checks live here one part at a time: :mod:`.metrics` today, dataset and
stage contracts to follow. This package depends only on
:mod:`espnet3.api`; it must never import :mod:`espnet3.components` or
:mod:`espnet3.systems` at module level. The declaration types
(:class:`~espnet3.api.inference.Field`, :class:`~espnet3.api.inference.Kind`)
and the field-declaration check
(:func:`~espnet3.api.inference.check.check_declaration`) stay in
:mod:`espnet3.api.inference`; this package only adds the checks specific
to each part.
"""
