"""Recipe-defined transformations shared by contract Kinds."""

from typing import Any, Callable

NORMALIZERS: dict[str, Callable[[Any], Any]] = {}


def register_normalizer(
    name: str, function: Callable[[Any], Any], *, replace: bool = False
) -> None:
    """Register a recipe transformation; the owning Kind checks its result.

    Args:
        name: The name config files use in ``normalize``.
        function: A callable from a Kind's checked value to a value of the same
            Kind.
        replace: Allow redefining a name already registered.

    Returns:
        None.

    Raises:
        TypeError: If ``function`` is not callable.
        ValueError: If ``name`` is taken and ``replace`` is false.

    Examples:
        >>> register_normalizer("shout", str.upper, replace=True)
        >>> NORMALIZERS["shout"]("hi")
        'HI'
    """
    if not callable(function):
        raise TypeError("normalizer must be callable")
    if name in NORMALIZERS and not replace:
        raise ValueError(f"normalizer {name!r} is already registered")
    NORMALIZERS[name] = function
