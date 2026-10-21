"""What a field's ``kind`` means: how a value is converted, checked and joined."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from espnet3.contract.field import Field


class Kind(ABC):
    """What a field's ``kind`` means: how a value is converted, checked and joined.

    A kind turns what a caller gives into what a hook receives, checks
    what a hook returns, and says how pieces of a stream join. The
    built-in kinds are ``audio``, ``text`` and ``segments``; a new
    modality - a conversation, video, a JSON document - is a subclass
    passed to :func:`register_kind`.

    Examples:
        >>> class Messages(Kind):
        ...     def check(self, value, field, model, *, output):
        ...         if not isinstance(value, list):
        ...             raise TypeError(f"{field.name} must be a list of turns")
        ...         return value
        >>> register_kind("messages", Messages())
        >>> Field("messages", "messages").kind
        'messages'
    """

    @abstractmethod
    def check(self, value: Any, field: Field, model: Any, *, output: bool) -> Any:
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

        Returns:
            ``value`` in the form this kind holds it.

        Examples:
            >>> from espnet3.contract import Field
            >>> from espnet3.contract.kinds.text import TextKind
            >>> TextKind().check("hi", Field("text", "text"), None, output=False)
            'hi'
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

        Examples:
            >>> from espnet3.contract.kinds.text import TextKind
            >>> TextKind().join("a", "b")
            'ab'
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not say how two pieces join; "
            "override join() to stream this kind"
        )

    def is_batch(self, value: Any, field: Field, model: Any) -> bool:
        """Tell one entry per sample from one value of this kind.

        A list that is not itself a valid value of the kind is a batch:
        a list of arrays is several utterances, a ``[rate, samples]`` pair
        is one; a list of turns is one conversation, a list of those is
        several. Override when a kind can say it more cheaply.

        Args:
            value: What the caller gave.
            field: The declaration the value is for.
            model: The instance, for what it knows (its ``sample_rate``).

        Returns:
            ``True`` when ``value`` is a list with one entry per sample.

        Examples:
            >>> from espnet3.contract import Field
            >>> from espnet3.contract.kinds.text import TextKind
            >>> TextKind().is_batch(["a", "b"], Field("text", "text"), None)
            True
            >>> TextKind().is_batch("a", Field("text", "text"), None)
            False
        """
        if not isinstance(value, list):
            return False
        try:
            self.check(value, field, model, output=False)
        except TypeError:
            return True
        return False

    def accepts(self, value: Any, field: Field) -> bool:
        """Tell whether ``value`` could hold this kind, with no model.

        For checking a value found with nothing to run - a dataset item -
        where :meth:`check` is not an option because it may need the
        model it is never given. Defaults to calling
        :meth:`check` with ``model=None`` and treating ``TypeError`` as
        "no"; override when a kind's :meth:`check` needs the model for
        something a model-free caller cannot supply (``audio`` needs a
        rate when none is carried).

        Args:
            value: The value to test.
            field: The declaration the value is for.

        Returns:
            ``True`` when ``check`` accepts ``value`` without a model.

        Examples:
            >>> from espnet3.contract import Field
            >>> from espnet3.contract.kinds.text import TextKind
            >>> TextKind().accepts("hi", Field("text", "text"))
            True
            >>> TextKind().accepts(7, Field("text", "text"))
            False
        """
        try:
            self.check(value, field, None, output=False)
        except TypeError:
            return False
        return True
