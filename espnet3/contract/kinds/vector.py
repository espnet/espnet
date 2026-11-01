"""The ``vector`` kind: a one-dimensional array of finite real numbers."""

from __future__ import annotations

import numpy as np

from espnet3.contract.kinds.base import Kind


class VectorKind(Kind):
    """``vector``: a 1-D array of finite real numbers; does not join.

    What a fixed-size embedding is: the speaker x-vector a multi-speaker TTS
    model is conditioned on, say. Anything array-like is accepted (a NumPy
    array, a torch tensor, a list of numbers) and returned as a NumPy array.
    The size is not checked here: it is the model's own concern (an ECAPA
    embedding has 192 values, a RawNet3 one 256), so a reusable kind cannot
    fix it.

    Examples:
        >>> from espnet3.contract import Field
        >>> field = Field("spembs", "vector")
        >>> VectorKind().check([0.0, 1.0], field, None, output=False)
        array([0., 1.])
    """

    def check(self, value, field, model, *, output):
        """Require a 1-D array of finite real numbers; return it as a NumPy array.

        Args:
            value: What the caller gave, or what the hook returned: a NumPy
                array, a torch tensor, or a sequence of numbers.
            field: The declaration, for the name in the error.
            model: Unused; a vector needs nothing from the model.
            output: Whether ``value`` is a hook's result.

        Returns:
            The values as a 1-D NumPy array (a tensor is moved to the CPU).

        Raises:
            TypeError: If ``value`` is not array-like, not one-dimensional,
                empty, or not real-numeric (``bool``, complex, strings and
                objects are refused).
            ValueError: If ``value`` holds a NaN or an infinity.

        Examples:
            >>> from espnet3.contract import Field
            >>> field = Field("spembs", "vector")
            >>> ones = np.ones(3, dtype=np.float32)
            >>> VectorKind().check(ones, field, None, output=False)
            array([1., 1., 1.], dtype=float32)
            >>> VectorKind().check(np.ones((2, 3)), field, None, output=True)
            Traceback (most recent call last):
            TypeError: 'spembs' returned with 2 dimensions, must be a 1-D numeric vector
        """
        where = "returned" if output else "given"
        if hasattr(value, "detach"):  # a torch tensor, without importing torch
            value = value.detach().cpu().numpy()
        try:
            vector = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, "
                "must be a 1-D numeric vector"
            ) from error
        if vector.ndim != 1:
            raise TypeError(
                f"{field.name!r} {where} with {vector.ndim} dimensions, "
                "must be a 1-D numeric vector"
            )
        if (
            not np.issubdtype(vector.dtype, np.number)
            or np.issubdtype(vector.dtype, np.complexfloating)
            or np.issubdtype(vector.dtype, np.bool_)
        ):
            raise TypeError(
                f"{field.name!r} {where} with dtype {vector.dtype}, "
                "must be a 1-D numeric vector"
            )
        if vector.size == 0:
            raise TypeError(
                f"{field.name!r} {where} empty, must hold at least one value"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"{field.name!r} {where} with NaN or Inf values")
        return vector
