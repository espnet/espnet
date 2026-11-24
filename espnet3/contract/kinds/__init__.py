"""The kinds a contract field can hold, and their registry.

``audio``, ``text``, ``segments``, ``number`` and ``vector`` are built in. A
new modality - a conversation, video, a JSON document - is a
:class:`Kind` subclass passed to :func:`register_kind`::

    from espnet3.contract import Field, Kind, register_kind

    class Messages(Kind):
        def check(self, value, field, model, *, output):
            if not isinstance(value, list):
                raise TypeError(f"{field.name} must be a list of turns")
            return value

    register_kind("messages", Messages())

"""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio, AudioKind
from espnet3.contract.kinds.base import Kind
from espnet3.contract.kinds.number import NumberKind
from espnet3.contract.kinds.segments import SegmentsKind
from espnet3.contract.kinds.text import TextKind
from espnet3.contract.kinds.vector import VectorKind

# What a field can hold, by the name a Field's ``kind`` gives.
KINDS: dict[str, Kind] = {
    "audio": AudioKind(),
    "text": TextKind(),
    "segments": SegmentsKind(),
    "number": NumberKind(),
    "vector": VectorKind(),
}


def register_kind(name: str, kind: Kind, *, replace: bool = False) -> None:
    """Add a kind under ``name``, so a :class:`Field` can declare it.

    Args:
        name: The name fields use, such as ``"messages"``.
        kind: The :class:`Kind` instance that converts and checks it.
        replace: Allow redefining a name already registered; off by
            default, so two systems cannot silently disagree on one.

    Returns:
        None.

    Raises:
        TypeError: If ``kind`` is not a :class:`Kind`.
        ValueError: If ``name`` is taken and ``replace`` is false.

    Examples:
        >>> register_kind("messages", Messages())
        >>> Field("messages", "messages").kind
        'messages'
    """
    if not isinstance(kind, Kind):
        raise TypeError(f"a kind is a Kind instance, not {type(kind).__name__}")
    if name in KINDS and not replace:
        raise ValueError(
            f"kind {name!r} is already registered; pass replace=True to redefine it"
        )
    KINDS[name] = kind


__all__ = [
    "KINDS",
    "Audio",
    "AudioKind",
    "Kind",
    "NumberKind",
    "SegmentsKind",
    "TextKind",
    "VectorKind",
    "register_kind",
]
