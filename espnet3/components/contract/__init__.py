"""Contract-checking code shared by what declares typed inputs and outputs.

The declaration types themselves (:class:`~espnet3.api.inference.Field`,
:class:`~espnet3.api.inference.Kind`) live in :mod:`espnet3.api.inference`;
this package holds only the checking code built on them, such as
:mod:`.metrics` for a metric's declared inputs/outputs.
"""

from __future__ import annotations
