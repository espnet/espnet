from typing import Mapping, Optional, Tuple, Type

from typeguard import typechecked

from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str_or_none


class ClassChoices:
    """Helper class to manage the options for variable objects and its configuration.

    Example:

    >>> class A:
    ...     def __init__(self, foo=3):  pass
    >>> class B:
    ...     def __init__(self, bar="aaaa"):  pass
    >>> choices = ClassChoices("var", dict(a=A, b=B), default="a")
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> choices.add_arguments(parser)
    >>> args = parser.parse_args(["--var", "a", "--var_conf", "foo=4")
    >>> args.var
    a
    >>> args.var_conf
    {"foo": 4}
    >>> class_obj = choices.get_class(args.var)
    >>> a_object = class_obj(**args.var_conf)

    """

    @typechecked
    def __init__(
        self,
        name: str,
        classes: Mapping[str, Type],
        type_check: Optional[Type] = None,
        default: Optional[str] = None,
        optional: bool = False,
    ):
        self.name = name
        self.base_type = type_check
        self.classes = {k.lower(): v for k, v in classes.items()}
        if "none" in self.classes or "nil" in self.classes or "null" in self.classes:
            raise ValueError('"none", "nil", and "null" are reserved.')
        if type_check is not None:
            for v in self.classes.values():
                if not issubclass(v, type_check):
                    raise ValueError(f"must be {type_check.__name__}, but got {v}")

        self.optional = optional
        self.default = default
        if default is None:
            self.optional = True

    def choices(self) -> Tuple[Optional[str], ...]:
        retval = tuple(self.classes)
        if self.optional:
            return retval + (None,)
        else:
            return retval

    @typechecked
    def get_class(self, name: Optional[str]) -> Optional[type]:
        if name is None or (self.optional and name.lower() == ("none", "null", "nil")):
            retval = None
        elif name.lower() in self.classes:
            class_obj = self.classes[name]
            retval = class_obj
        else:
            raise ValueError(
                f"--{self.name} must be one of {self.choices()}: "
                f"--{self.name} {name.lower()}"
            )

        return retval

    def add_arguments(self, parser):
        parser.add_argument(
            f"--{self.name}",
            type=lambda x: str_or_none(x.lower()),
            default=self.default,
            choices=self.choices(),
            help=f"The {self.name} type",
        )
        parser.add_argument(
            f"--{self.name}_conf",
            action=NestedDictAction,
            default=dict(),
            help=f"The keyword arguments for {self.name}",
        )
