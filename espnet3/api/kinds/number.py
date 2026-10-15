"""The ``number`` kind: a real number (int, float or NumPy scalar)."""

import numbers

import numpy as np

from espnet3.api.kinds.base import Kind


class NumberKind(Kind):
    """``number``: a real number, not ``bool``; does not join.

    What a metric's declared outputs hold: one scalar per result key, such
    as ``{"WER": 4.3}``.

    Examples:
        >>> NumberKind().check(4.3, Field("WER", "number"), model, output=True)
        4.3
    """

    def check(self, value, field, model, *, output):
        """Require a real number, excluding ``bool``.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration, for the name in the error.
            model: Unused; a number needs nothing from the model.
            output: Whether ``value`` is a hook's result.

        Returns:
            ``value`` itself.

        Raises:
            TypeError: If ``value`` is not a real number (``numbers.Real``),
                or is a ``bool`` or ``np.bool_``.

        Examples:
            >>> NumberKind().check(4.3, Field("WER", "number"), model, output=True)
            4.3
            >>> NumberKind().check(
            ...     np.float32(4.3), Field("WER", "number"), model, output=True
            ... )
            np.float32(4.3)
            >>> NumberKind().check(True, Field("WER", "number"), model, output=True)
            Traceback (most recent call last):
            TypeError: 'WER' returned as bool, must be a real number
        """
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, "
                "must be a real number"
            )
        return value
