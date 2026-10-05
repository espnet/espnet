"""The declaration types a contract is built from: :class:`Field` and :class:`Kind`.

Shared by the inference contract and by whatever else declares typed
inputs and outputs - metrics, a dataset's columns.
"""

from __future__ import annotations

from espnet3.components.contract.check import check_declaration
from espnet3.components.contract.field import Field
from espnet3.components.contract.kinds import (
    KINDS,
    Audio,
    AudioKind,
    Kind,
    NumberKind,
    SegmentsKind,
    TextKind,
    register_kind,
)

__all__ = [
    "KINDS",
    "Audio",
    "AudioKind",
    "Field",
    "Kind",
    "NumberKind",
    "SegmentsKind",
    "TextKind",
    "check_declaration",
    "register_kind",
]
