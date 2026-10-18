"""Recipe-defined transformations shared by contract Kinds."""

from typing import Any, Callable

NORMALIZERS: dict[str, Callable[[Any], Any]] = {}


def register_normalizer(
    name: str, function: Callable[[Any], Any], *, replace: bool = False
) -> None:
    """Register a recipe transformation; the owning Kind checks its result."""
    if not callable(function):
        raise TypeError("normalizer must be callable")
    if name in NORMALIZERS and not replace:
        raise ValueError(f"normalizer {name!r} is already registered")
    NORMALIZERS[name] = function
