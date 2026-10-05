"""The ``number`` kind: an ``int`` or ``float``, for a metric's result."""

from __future__ import annotations

from espnet3.components.contract.kinds.base import Kind


class NumberKind(Kind):
    """``number``: an ``int`` or ``float`` (not ``bool``); does not join.

    What a metric's declared outputs hold: one scalar per result key, such
    as ``{"WER": 4.3}``.
    """

    def check(self, value, field, model, *, output):
        """Require an ``int`` or ``float``, excluding ``bool``.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration, for the name in the error.
            model: Unused; a number needs nothing from the model.
            output: Whether ``value`` is a hook's result.

        Returns:
            ``value`` itself.

        Raises:
            TypeError: If ``value`` is not an ``int``/``float``, or is a
                ``bool`` (a `bool` is technically an `int` in Python, but
                was never meant as a metric score).

        Examples:
            >>> NumberKind().check(4.3, Field("WER", "number"), model, output=True)
            4.3
            >>> NumberKind().check(True, Field("WER", "number"), model, output=True)
            Traceback (most recent call last):
            TypeError: 'WER' returned as bool, must be int or float
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, "
                "must be int or float"
            )
        return value
