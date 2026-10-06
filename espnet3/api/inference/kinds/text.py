"""The ``text`` kind: a ``str``; pieces of a stream append."""

from __future__ import annotations

from espnet3.api.inference.kinds.base import Kind


class TextKind(Kind):
    """``text``: a ``str``; pieces append."""

    def check(self, value, field, model, *, output):
        """Require a ``str``; return it unchanged.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration, for the name in the error.
            model: Unused; text needs nothing from the model.
            output: Whether ``value`` is a hook's result.

        Returns:
            ``value`` itself.

        Raises:
            TypeError: If ``value`` is not a ``str``.

        Examples:
            >>> TextKind().check("hello", Field("text", "text"), model, output=False)
            'hello'
            >>> TextKind().check(7, Field("text", "text"), model, output=True)
            Traceback (most recent call last):
            TypeError: 'text' returned as int, must be str
        """
        if not isinstance(value, str):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, must be str"
            )
        return value

    def join(self, first, second):
        """Append the text."""
        return first + second
