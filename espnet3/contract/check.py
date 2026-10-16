"""Shared validation for a class's or an instance's field declarations."""

from __future__ import annotations

from espnet3.contract.field import Field


def check_declaration(
    obj, inputs_attr: str = "inputs", outputs_attr: str = "outputs"
) -> None:
    """Raise ``TypeError`` unless ``obj`` declares its fields correctly.

    Shared by every interface declared with :class:`Field`: a tuple of fields
    with distinct names, required ones before optional ones, and no
    optional output.

    Args:
        obj: The class to check, or an instance whose own attributes (set
            in its ``__init__``) declare fields its class does not.
        inputs_attr: The name of the inputs attribute to check.
        outputs_attr: The name of the outputs attribute to check.

    Raises:
        TypeError: With the rule that was broken.

    Examples:
        >>> from espnet3.contract import Field
        >>> class Bad:
        ...     inputs = (Field("prompt", "text", optional=True),
        ...               Field("speech", "audio"))
        ...     outputs = (Field("text", "text"),)
        >>> check_declaration(Bad)
        Traceback (most recent call last):
        TypeError: Bad.inputs must list required fields before optional ones, ...
    """
    cls = obj if isinstance(obj, type) else type(obj)
    for attr in (inputs_attr, outputs_attr):
        fields = getattr(obj, attr, None)
        if not isinstance(fields, tuple) or not all(
            isinstance(f, Field) for f in fields
        ):
            raise TypeError(f"{cls.__qualname__}.{attr} must be a tuple of Field")
        if not fields:
            raise TypeError(f"{cls.__qualname__}.{attr} must name at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise TypeError(f"{cls.__qualname__}.{attr} repeats a name: {names}")
    optional = [f.optional for f in getattr(obj, inputs_attr)]
    if optional != sorted(optional):
        raise TypeError(
            f"{cls.__qualname__}.{inputs_attr} must list required fields before "
            "optional ones, so a call by position means one thing"
        )
    for f in getattr(obj, outputs_attr):
        if f.optional:
            raise TypeError(f"{cls.__qualname__}: {outputs_attr} cannot be optional")
