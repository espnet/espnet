"""Checking a dataset's declared item fields."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from espnet3.api.inference import KINDS, Field


class DatasetContractError(ValueError):
    """A dataset item does not match its declared fields.

    Examples:
        >>> raise DatasetContractError("item lacks declared field 'text'")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: item lacks ...
    """


def require_fields(obj, attr: str) -> None:
    """Raise because ``obj`` declares no ``attr``.

    Every dataset must declare its item fields - as a class attribute, or
    set on ``self`` for one that depends on its own configuration; there
    is no undeclared fallback.

    Args:
        obj: The class to check, or an instance whose own attributes
            (set in its ``__init__``) declare a contract its class does
            not.
        attr: The attribute name, such as ``"fields"``.

    Examples:
        >>> class Undeclared:
        ...     pass
        >>> require_fields(Undeclared, "fields")
        Traceback (most recent call last):
        TypeError: Undeclared does not declare fields; declare `fields` ...
    """
    cls = obj if isinstance(obj, type) else type(obj)
    raise TypeError(
        f"{cls.__qualname__} does not declare {attr}; declare `{attr}` "
        "(a class attribute, or set on `self`)."
    )


def check_fields(obj, attr: str) -> Optional[Tuple[Field, ...]]:
    """Return ``obj.<attr>`` if well-formed, or ``None`` if undeclared.

    A dataset's ``fields`` is one standalone tuple: a tuple of ``Field``,
    non-empty, with distinct names. ``None`` lets a caller decide whether
    an undeclared ``attr`` is allowed, or not (call :func:`require_fields`).

    Args:
        obj: The class to check, or an instance whose own attributes
            (set in its ``__init__``) declare a contract its class does
            not.
        attr: The attribute name, such as ``"fields"``.

    Raises:
        TypeError: ``obj.<attr>`` is declared but malformed.

    Examples:
        >>> class Good:
        ...     fields = (Field("speech", "audio"), Field("text", "text"))
        >>> check_fields(Good, "fields")[0].name
        'speech'
        >>> class Bad:
        ...     fields = (Field("text", "text"), Field("text", "text"))
        >>> check_fields(Bad, "fields")
        Traceback (most recent call last):
        TypeError: Bad.fields repeats a name: ['text', 'text']
    """
    cls = obj if isinstance(obj, type) else type(obj)
    fields = getattr(obj, attr, None)
    if fields is None:
        return None
    if not isinstance(fields, tuple) or not all(isinstance(f, Field) for f in fields):
        raise TypeError(f"{cls.__qualname__}.{attr} must be a tuple of Field")
    if not fields:
        raise TypeError(f"{cls.__qualname__}.{attr} must name at least one field")
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise TypeError(f"{cls.__qualname__}.{attr} repeats a name: {names}")
    return fields


def check_item(fields: Tuple[Field, ...], item: Any, where: str) -> None:
    """Raise unless ``item`` has every declared field, each holding its kind.

    Args:
        fields: The declared item fields.
        item: One dataset sample, taken from the recipe's own
            ``__getitem__``, before any transform or preprocessor.
        where: Where ``item`` came from, for the error.

    Raises:
        DatasetContractError: ``item`` is not a mapping, lacks a required
            declared field, or a field's value does not match its kind. An
            undeclared extra key is not an error.

    Examples:
        >>> import numpy as np
        >>> fields = (Field("speech", "audio"), Field("text", "text"))
        >>> item = {"speech": np.zeros(16000, dtype=np.float32), "text": "hi"}
        >>> check_item(fields, item, "train[0]")
        >>> check_item(fields, {"speech": item["speech"]}, "train[0]")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: train[0]: ...
    """
    if not isinstance(item, Mapping):
        raise DatasetContractError(
            f"{where}: item must be a dict, got {type(item).__name__}"
        )
    for f in fields:
        if f.name not in item:
            if f.optional:
                continue
            raise DatasetContractError(
                f"{where}: item lacks declared field {f.name!r} ({f.kind}); "
                f"keys are {sorted(item)}"
            )
        if not KINDS[f.kind].accepts(item[f.name], f):
            raise DatasetContractError(
                f"{where}: field {f.name!r} is declared {f.kind} but the item "
                f"holds {type(item[f.name]).__name__}"
            )


__all__ = [
    "DatasetContractError",
    "check_fields",
    "check_item",
    "require_fields",
]
