"""What a field's ``kind`` means: how a value is converted, checked and joined."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from espnet3.api.inference.base import InferenceAPI
    from espnet3.api.inference.field import Field


class Kind(ABC):
    """What a field's ``kind`` means: how a value is converted, checked and joined.

    A kind turns what a caller gives into what a hook receives, checks
    what a hook returns, and says how pieces of a stream join. The
    built-in kinds are ``audio``, ``text`` and ``segments``; a new
    modality - a conversation, a multichannel signal, video, a JSON
    document - is a subclass passed to :func:`register_kind`, and needs no
    change to :class:`InferenceAPI`.

    Examples:
        >>> class Messages(Kind):
        ...     def check(self, value, field, model, *, output):
        ...         if not isinstance(value, list):
        ...             raise TypeError(f"{field.name} must be a list of turns")
        ...         return value
        >>> register_kind("messages", Messages())
        >>> Field("messages", "messages")
        Field(name='messages', kind='messages', label='Messages', optional=False)
    """

    @abstractmethod
    def check(
        self, value: Any, field: Field, model: "InferenceAPI", *, output: bool
    ) -> Any:
        """Return ``value`` as this kind holds it, or raise ``TypeError``.

        The kind decides the form; ``field`` supplies the name for the
        error and any per-field detail a kind may read off its declaration.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration the value is for.
            model: The instance, for what it knows (its ``sample_rate``).
            output: Whether the value is a hook's result rather than a
                caller's argument; a kind may accept different forms for
                each.
        """

    def join(self, first: Any, second: Any) -> Any:
        """Join two consecutive pieces of one field, for :func:`gather`.

        Each kind says how its pieces join - text appends, audio
        concatenates in time - and a kind that does not say cannot be
        streamed: there is no default, since ``+`` on two arrays would
        silently add them.

        Args:
            first: The piece gathered so far.
            second: The piece that follows it.

        Returns:
            The two as one piece.

        Raises:
            NotImplementedError: Always, unless the kind overrides it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not say how two pieces join; "
            "override join() to stream this kind"
        )

    def is_batch(self, value: Any, field: Field, model: "InferenceAPI") -> bool:
        """Tell one entry per sample from one value of this kind.

        A list that is not itself a valid value of the kind is a batch:
        a list of arrays is several utterances, a ``[rate, samples]`` pair
        is one; a list of turns is one conversation, a list of those is
        several. Override when a kind can say it more cheaply.
        """
        if not isinstance(value, list):
            return False
        try:
            self.check(value, field, model, output=False)
        except TypeError:
            return True
        return False
