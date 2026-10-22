"""The declaration of one input or output of a system."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Optional

from espnet3.contract.kinds import KINDS


@dataclass(frozen=True)
class Field:
    """One input or output of a system.

    A system lists these in its ``inputs`` and ``outputs``; the base class
    binds arguments to them, converts and checks values by ``kind``, and a
    front end builds its widgets or command-line arguments from them.

    Args:
        name: The keyword the value is passed or returned as, such as
            ``speech`` or ``text``. Must be a Python identifier.
        kind: A name in :data:`~espnet3.contract.kinds.KINDS`. Decides
            how a value is converted
            and checked (``audio`` becomes an :class:`Audio` at the model's
            rate, ``text`` must be a ``str``) and, for a front end, which
            widget or argument type shows it.
        label: What a page calls the field. Defaults to the name with
            underscores spaced and the first letter capitalised.
        optional: An input the caller may leave out; the hook then does
            not receive it. Outputs are never optional.
        channels: Required audio channel count, or ``None`` (default) to
            preserve every channel. Channel selection and averaging are
            explicit audio normalizers.
        options: Kind-specific settings, such as ``sample_rate`` or
            ``normalize``. The registered Kind validates these options.

    Raises:
        ValueError: If ``kind`` is not registered in :data:`KINDS`,
            ``name`` is not an identifier, or ``channels`` is below 1.

    Examples:
        >>> Field("speech", "audio").kind
        'audio'
        >>> Field("reference_speech", "audio", optional=True).label
        'Reference speech'
        >>> Field("text", "text", "Transcription").label
        'Transcription'
        >>> Field("mixture", "audio", channels=None).channels   # a multichannel model
        >>> Field("speech", "audio").channels is None
        True
    """

    name: str
    kind: str
    label: str = ""
    optional: bool = False
    channels: Optional[int] = None
    options: Mapping[str, Any] = dataclass_field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Check the kind and the name, and fill in the label."""
        if self.kind not in KINDS:
            raise ValueError(
                f"Field {self.name!r} has kind {self.kind!r}; "
                f"known kinds are {sorted(KINDS)}"
            )
        if not isinstance(self.options, Mapping):
            raise TypeError("Field.options must be a mapping")
        KINDS[self.kind].validate_options(self.options)
        if not self.name.isidentifier():
            raise ValueError(f"Field name {self.name!r} must be a Python identifier")
        if self.channels is not None and self.channels < 1:
            raise ValueError(f"Field {self.name!r}: channels must be None or >= 1")
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").capitalize())
