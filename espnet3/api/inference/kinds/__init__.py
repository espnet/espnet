"""The kinds a field can hold, and the registry a new one is added to.

``audio``, ``text`` and ``segments`` are built in. A new modality - a
conversation, a multichannel signal, video, a JSON document - is a
:class:`BaseKind` subclass passed to :func:`register_kind`, from a system
or from a recipe's own ``src/``; :class:`~espnet3.api.inference.base.BaseInference`
needs no change for it::

    from espnet3.api.inference import BaseInference, BaseKind, Field, register_kind

    class Messages(BaseKind):
        def check(self, value, field, model, *, output):
            if not isinstance(value, list):
                raise TypeError(f"{field.name} must be a list of turns")
            return value

    register_kind("messages", Messages())

    class Inference(BaseInference):
        inputs = (Field("messages", "messages"),)
        ...
"""

from __future__ import annotations

from espnet3.api.inference.kinds.audio import Audio, AudioKind
from espnet3.api.inference.kinds.base import BaseKind
from espnet3.api.inference.kinds.segments import SegmentsKind
from espnet3.api.inference.kinds.text import TextKind

# What a field can hold, by the name a Field's ``kind`` gives.
KINDS: dict[str, BaseKind] = {
    "audio": AudioKind(),
    "text": TextKind(),
    "segments": SegmentsKind(),
}


def register_kind(name: str, kind: BaseKind, *, replace: bool = False) -> None:
    """Add a kind under ``name``, so a :class:`Field` can declare it.

    Args:
        name: The name fields use, such as ``"messages"``.
        kind: The :class:`BaseKind` instance that converts and checks it.
        replace: Allow redefining a name already registered; off by
            default, so two systems cannot silently disagree on one.

    Raises:
        TypeError: If ``kind`` is not a :class:`BaseKind`.
        ValueError: If ``name`` is taken and ``replace`` is false.

    Examples:
        >>> register_kind("messages", Messages())
        >>> Field("messages", "messages")
        Field(name='messages', kind='messages', label='Messages', optional=False)
    """
    if not isinstance(kind, BaseKind):
        raise TypeError(f"a kind is a BaseKind instance, not {type(kind).__name__}")
    if name in KINDS and not replace:
        raise ValueError(
            f"kind {name!r} is already registered; pass replace=True to redefine it"
        )
    KINDS[name] = kind


__all__ = [
    "KINDS",
    "Audio",
    "AudioKind",
    "BaseKind",
    "SegmentsKind",
    "TextKind",
    "register_kind",
]
