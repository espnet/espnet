"""Dataset item contracts expressed with ordinary Python runtime types."""

from collections.abc import Mapping


class DatasetContractError(ValueError):
    """A dataset declaration or item violates its contract."""


def check_fields(obj, attr="fields"):
    """Require a nonempty name-to-type mapping, accepting tuples of types.

    Include ``type(None)`` among the accepted types to make a key optional.
    Extra item keys are allowed. No inference declarations are involved.
    """
    fields = getattr(obj, attr, None)
    if not isinstance(fields, Mapping) or not fields:
        raise DatasetContractError(
            f"{type(obj).__name__}.{attr} must map names to types"
        )
    for name, expected in fields.items():
        types = expected if isinstance(expected, tuple) else (expected,)
        if (
            not isinstance(name, str)
            or not name
            or not types
            or not all(isinstance(t, type) for t in types)
        ):
            raise DatasetContractError(f"{attr}[{name!r}] must declare Python types")
    return fields


def check_item(fields, item, *, where="dataset"):
    """Check required keys and value types in one dataset item."""
    if not isinstance(item, Mapping):
        raise DatasetContractError(f"{where} must return a mapping")
    for name, expected in fields.items():
        if name not in item:
            if isinstance(None, expected):
                continue
            raise DatasetContractError(f"{where} is missing {name!r}")
        if not isinstance(item[name], expected):
            raise DatasetContractError(
                f"{where}[{name!r}] must match {expected}, "
                f"got {type(item[name]).__name__}"
            )
