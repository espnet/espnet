"""The base class a recipe's own dataset class promises its items against."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Mapping, Tuple

from torch.utils.data import Dataset as TorchDataset

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import (
    check_fields,
    check_item,
    require_fields,
)


class BaseDataset(TorchDataset, ABC):
    """What a recipe's dataset promises: the keys and kinds of one item.

    A subclass declares ``fields`` - a class attribute when every instance's
    item shape is the same, or set on ``self`` in ``__init__`` when it
    depends on the dataset's own arguments - and implements
    ``__getitem__``. The first call to ``__getitem__`` on each instance is
    checked against the declaration; later calls are not, so checking adds
    no cost once a dataset is known good.

    Examples:
        >>> class MiniDataset(BaseDataset):
        ...     fields = (Field("utt_id", "text"), Field("text", "text"))
        ...     def __init__(self):
        ...         self.items = [{"utt_id": "u1", "text": "hi"}]
        ...     def __len__(self):
        ...         return len(self.items)
        ...     def __getitem__(self, index):
        ...         return self.items[index]
        >>> MiniDataset()[0]["text"]
        'hi'
    """

    fields: ClassVar[Tuple[Field, ...]]

    def __init_subclass__(cls, **kwargs):
        """Validate class-level ``fields`` and wrap ``__getitem__`` to check once."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "fields"):
            check_fields(cls, "fields")
        if not getattr(cls.__getitem__, "__wrapped_contract__", False):
            cls.__getitem__ = _checked_once(cls.__getitem__)

    @abstractmethod
    def __getitem__(self, index) -> Mapping[str, Any]:
        """Return one item, as a mapping holding every declared field."""


def _checked_once(getitem: Callable) -> Callable:
    """Wrap ``getitem`` to check its first result against ``self.fields``."""

    @functools.wraps(getitem)
    def wrapper(self, index):
        item = getitem(self, index)
        if not getattr(self, "_fields_checked", False):
            fields = check_fields(self, "fields")
            if fields is None:
                require_fields(self, "fields")
            check_item(fields, item, where=type(self).__qualname__)
            self._fields_checked = True
        return item

    wrapper.__wrapped_contract__ = True
    return wrapper
