"""The ``segments`` kind: aligned spans with text, start and end in seconds."""

from __future__ import annotations

from typing import Mapping

from espnet3.api.inference.kinds.base import BaseKind


class SegmentsKind(BaseKind):
    """``segments``: dicts with ``text``, ``start`` and ``end`` in seconds.

    The shape ``espnet align`` prints; ``score`` is optional. Pieces append.
    """

    def check(self, value, field, model, *, output):
        """Require a list of dicts with ``text``, ``start`` and ``end``.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration, for the name in the error.
            model: Unused.
            output: Whether ``value`` is a hook's result.

        Returns:
            ``value`` itself; extra keys such as ``score`` pass through.

        Raises:
            TypeError: If ``value`` is not a list, or an element lacks one
                of the three keys.

        Examples:
            >>> SegmentsKind().check(
            ...     [{"text": "hi", "start": 0.0, "end": 0.4, "score": 0.9}],
            ...     Field("segments", "segments"), model, output=True)
            [{'text': 'hi', 'start': 0.0, 'end': 0.4, 'score': 0.9}]
            >>> SegmentsKind().check(
            ...     [(0.0, 0.4)], Field("segments", "segments"), model, output=True)
            Traceback (most recent call last):
            TypeError: 'segments' returned must be a list of dicts with text, ...
        """
        if not isinstance(value, list) or not all(
            isinstance(s, Mapping) and {"text", "start", "end"} <= set(s) for s in value
        ):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} must be a list of dicts "
                "with text, start and end"
            )
        return value
