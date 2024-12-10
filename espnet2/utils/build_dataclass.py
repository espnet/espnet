import argparse
import dataclasses


def build_dataclass(dataclass, args: argparse.Namespace):
    """
        Helper function to build a dataclass from an argparse.Namespace object.

    This function takes a dataclass and an argparse.Namespace instance, extracts the
    values corresponding to the dataclass fields from the namespace, and constructs an
    instance of the dataclass. If any required field is missing from the namespace, a
    ValueError is raised.

    Args:
        dataclass: The dataclass type to be instantiated.
        args (argparse.Namespace): An instance of argparse.Namespace containing
            the arguments.

    Returns:
        An instance of the specified dataclass populated with values from 'args'.

    Raises:
        ValueError: If 'args' does not contain a value for any required field of
            the dataclass.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     learning_rate: float
        ...     batch_size: int
        ...
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--learning_rate', type=float, required=True)
        >>> parser.add_argument('--batch_size', type=int, required=True)
        >>> args = parser.parse_args(['--learning_rate', '0.001', '--batch_size', '32'])
        >>> config = build_dataclass(Config, args)
        >>> print(config)
        Config(learning_rate=0.001, batch_size=32)
    """
    kwargs = {}
    for field in dataclasses.fields(dataclass):
        if not hasattr(args, field.name):
            raise ValueError(
                f"args doesn't have {field.name}. You need to set it to ArgumentsParser"
            )
        kwargs[field.name] = getattr(args, field.name)
    return dataclass(**kwargs)
