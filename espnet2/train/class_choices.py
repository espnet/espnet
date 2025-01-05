from typing import Mapping, Optional, Tuple, Type

from typeguard import typechecked

from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str_or_none


class ClassChoices:
    """
    Helper class to manage the options for variable objects and their configuration.

    This class allows the definition of a set of classes that can be chosen
    dynamically at runtime, along with optional configurations for those classes.
    It also provides methods to add command-line argument parsing support for
    selecting the class and passing configuration parameters.

    Attributes:
        name (str): The name of the variable being managed.
        base_type (Optional[Type]): The base type that all classes must inherit from.
        classes (Mapping[str, Type]): A mapping of class names to class types.
        optional (bool): Indicates if the choice is optional.
        default (Optional[str]): The default class name if none is specified.

    Args:
        name (str): The name of the variable.
        classes (Mapping[str, Type]): A mapping of class names to class types.
        type_check (Optional[Type]): A base class type that the classes must inherit from.
        default (Optional[str]): The default class name to use if none is provided.
        optional (bool): Whether the choice of class is optional.

    Returns:
        None

    Raises:
        ValueError: If "none", "nil", or "null" are included in class names or if
                    a class does not inherit from the specified base type.

    Examples:
        >>> class A:
        ...     def __init__(self, foo=3):  pass
        >>> class B:
        ...     def __init__(self, bar="aaaa"):  pass
        >>> choices = ClassChoices("var", dict(a=A, b=B), default="a")
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> choices.add_arguments(parser)
        >>> args = parser.parse_args(["--var", "a", "--var_conf", "foo=4"])
        >>> args.var
        'a'
        >>> args.var_conf
        {'foo': 4}
        >>> class_obj = choices.get_class(args.var)
        >>> a_object = class_obj(**args.var_conf)

    Note:
        The class names are case insensitive and are stored in lowercase.

    Todo:
        - Consider adding more robust type checking or validation for configurations.
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
        """
                Helper class to manage the options for variable objects and their configuration.

        This class provides a way to define a set of choices for a variable, allowing
        for the selection of classes based on user input. It also facilitates the
        configuration of those classes through keyword arguments.

        Example:

        >>> class A:
        ...     def __init__(self, foo=3): pass
        >>> class B:
        ...     def __init__(self, bar="aaaa"): pass
        >>> choices = ClassChoices("var", dict(a=A, b=B), default="a")
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> choices.add_arguments(parser)
        >>> args = parser.parse_args(["--var", "a", "--var_conf", "foo=4"])
        >>> args.var
        'a'
        >>> args.var_conf
        {'foo': 4}
        >>> class_obj = choices.get_class(args.var)
        >>> a_object = class_obj(**args.var_conf)

        Attributes:
            name (str): The name of the variable.
            base_type (Optional[Type]): The base type for type checking.
            classes (Mapping[str, Type]): A mapping of class names to class types.
            optional (bool): Indicates if the choice is optional.
            default (Optional[str]): The default choice if none is provided.

        Args:
            name (str): The name of the variable.
            classes (Mapping[str, Type]): A mapping of strings to class types.
            type_check (Optional[Type]): A type to check against the classes.
            default (Optional[str]): The default choice for the variable.
            optional (bool): If True, the choice can be omitted.

        Raises:
            ValueError: If "none", "nil", or "null" is in classes or if a class
                does not subclass the specified type_check.
        """
        retval = tuple(self.classes)
        if self.optional:
            return retval + (None,)
        else:
            return retval

    @typechecked
    def get_class(self, name: Optional[str]) -> Optional[type]:
        """
            Retrieve the class associated with the given name.

        This method looks up the class in the internal mapping of classes and
        returns the corresponding class type. If the name is not found or is
        invalid, it raises a ValueError.

        Args:
            name (Optional[str]): The name of the class to retrieve. It should
                match one of the keys in the `classes` mapping, case-insensitively.

        Returns:
            Optional[type]: The class associated with the given name, or None
                if the name is None or represents a null value.

        Raises:
            ValueError: If the name is not one of the valid options in
                `choices()`.

        Examples:
            >>> choices = ClassChoices("var", dict(a=A, b=B), default="a")
            >>> class_obj = choices.get_class("a")
            >>> print(class_obj)
            <class '__main__.A'>

            >>> class_obj = choices.get_class("b")
            >>> print(class_obj)
            <class '__main__.B'>

            >>> class_obj = choices.get_class("none")
            >>> print(class_obj)
            None

            >>> choices.get_class("invalid")  # Raises ValueError
        """
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
        """
            Adds command-line arguments for the specified variable and its configuration.

        This method integrates with an argparse parser to allow users to specify
        the type of class they wish to instantiate along with its configuration
        parameters.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the
                arguments will be added.

        Examples:
            >>> import argparse
            >>> choices = ClassChoices("var", dict(a=A, b=B), default="a")
            >>> parser = argparse.ArgumentParser()
            >>> choices.add_arguments(parser)
            >>> args = parser.parse_args(["--var", "a", "--var_conf", "foo=4"])
            >>> args.var
            'a'
            >>> args.var_conf
            {'foo': 4}

        Note:
            The configuration parameters are expected to be provided as a
            dictionary. The `NestedDictAction` is used to parse these arguments
            appropriately.
        """
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
