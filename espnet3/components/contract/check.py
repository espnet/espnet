"""Shared validation for a class's or an instance's field declarations."""

from __future__ import annotations


def _qualname(obj: object) -> str:
    """Return ``obj``'s own qualified name, or its type's if it has none.

    ``obj`` is usually a class (``cls.__qualname__`` is its own), but a
    contract may instead be checked on an instance (one whose declaration
    comes from its own ``__init__``, not its class) - that has no
    ``__qualname__`` of its own, so its type's is used instead.
    """
    return getattr(obj, "__qualname__", None) or type(obj).__qualname__


def check_declaration(
    cls: type, inputs_attr: str = "inputs", outputs_attr: str = "outputs"
) -> None:
    """Raise ``TypeError`` unless ``cls`` declares its fields correctly.

    Shared by every contract built on :class:`Field`: a tuple of fields
    with distinct names, required ones before optional ones, and no
    optional output.

    Args:
        cls: The class to check, or an instance whose own attributes (set
            in its ``__init__``) declare a contract its class does not.
        inputs_attr: The name of the inputs attribute to check.
        outputs_attr: The name of the outputs attribute to check.

    Raises:
        TypeError: With the rule that was broken.

    Note:
        Imports :class:`Field` lazily (inside the function body, not at
        module load time): :class:`InferenceAPI` imports this module to
        share the check, so a module-level import of
        :mod:`espnet3.api.inference` here would import it back before it
        finishes initializing.

    Examples:
        >>> from espnet3.api.inference import Field
        >>> class Bad:
        ...     inputs = (Field("prompt", "text", optional=True),
        ...               Field("speech", "audio"))
        ...     outputs = (Field("text", "text"),)
        >>> check_declaration(Bad)
        Traceback (most recent call last):
        TypeError: Bad.inputs must list required fields before optional ones, ...
    """
    from espnet3.api.inference.field import Field

    name = _qualname(cls)
    for attr in (inputs_attr, outputs_attr):
        fields = getattr(cls, attr, None)
        if not isinstance(fields, tuple) or not all(
            isinstance(f, Field) for f in fields
        ):
            raise TypeError(f"{name}.{attr} must be a tuple of Field")
        if not fields:
            raise TypeError(f"{name}.{attr} must name at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise TypeError(f"{name}.{attr} repeats a name: {names}")
    optional = [f.optional for f in getattr(cls, inputs_attr)]
    if optional != sorted(optional):
        raise TypeError(
            f"{name}.{inputs_attr} must list required fields before "
            "optional ones, so a call by position means one thing"
        )
    for f in getattr(cls, outputs_attr):
        if f.optional:
            raise TypeError(f"{name}: {outputs_attr} cannot be optional")
