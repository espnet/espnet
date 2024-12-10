import collections
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text


class FloatRandomGenerateDataset(collections.abc.Mapping):
    """
    Generate float arrays from a shape specification file.

    This class reads a file containing the shapes of the arrays to be generated.
    The file should be formatted as follows:

        shape.txt
        uttA 123,83
        uttB 34,83

    Each line corresponds to a unique identifier and the dimensions of the
    float array that will be generated.

    Examples:
        >>> dataset = FloatRandomGenerateDataset("shape.txt")
        >>> array = dataset["uttA"]
        >>> assert array.shape == (123, 83)
        >>> array = dataset["uttB"]
        >>> assert array.shape == (34, 83)

    Attributes:
        utt2shape (dict): A mapping from utterance identifiers to their shapes.
        dtype (np.dtype): The data type of the generated arrays.

    Args:
        shape_file (Union[Path, str]): The path to the shape specification file.
        dtype (Union[str, np.dtype], optional): The data type of the generated
            arrays. Defaults to "float32".
        loader_type (str, optional): The type of loader to use for reading the
            shape file. Defaults to "csv_int".

    Raises:
        FileNotFoundError: If the specified shape file does not exist.
        ValueError: If the shape file is improperly formatted.
    """

    @typechecked
    def __init__(
        self,
        shape_file: Union[Path, str],
        dtype: Union[str, np.dtype] = "float32",
        loader_type: str = "csv_int",
    ):
        shape_file = Path(shape_file)
        self.utt2shape = load_num_sequence_text(shape_file, loader_type)
        self.dtype = np.dtype(dtype)

    def __iter__(self):
        return iter(self.utt2shape)

    def __len__(self):
        return len(self.utt2shape)

    def __getitem__(self, item) -> np.ndarray:
        shape = self.utt2shape[item]
        return np.random.randn(*shape).astype(self.dtype)


class IntRandomGenerateDataset(collections.abc.Mapping):
    """
        Generate integer arrays from a shape definition file.

    This class generates random integer arrays based on the shapes defined in a
    specified text file. The text file should list utterances and their respective
    shapes in a comma-separated format. The generated integers will be within the
    specified range defined by `low` and `high`.

    Attributes:
        low (int): The lower bound for the random integers.
        high (int): The upper bound for the random integers.
        dtype (np.dtype): The data type of the generated integers.
        utt2shape (dict): A mapping from utterance identifiers to their shapes.

    Args:
        shape_file (Union[Path, str]): The path to the shape definition file.
        low (int): The minimum value of the random integers.
        high (int, optional): The maximum value of the random integers. If not
            specified, defaults to None.
        dtype (Union[str, np.dtype], optional): The data type of the output
            integers. Defaults to "int64".
        loader_type (str, optional): The method used to load the shape file.
            Defaults to "csv_int".

    Returns:
        np.ndarray: A random integer array of the specified shape.

    Examples:
        shape.txt
        uttA 123,83
        uttB 34,83

        >>> dataset = IntRandomGenerateDataset("shape.txt", low=0, high=10)
        >>> array = dataset["uttA"]
        >>> assert array.shape == (123, 83)
        >>> array = dataset["uttB"]
        >>> assert array.shape == (34, 83)

    Note:
        The `high` parameter must be greater than `low` if specified.

    Raises:
        ValueError: If `high` is not specified and `low` is greater than or equal
        to `high`.
    """

    @typechecked
    def __init__(
        self,
        shape_file: Union[Path, str],
        low: int,
        high: int = None,
        dtype: Union[str, np.dtype] = "int64",
        loader_type: str = "csv_int",
    ):
        shape_file = Path(shape_file)
        self.utt2shape = load_num_sequence_text(shape_file, loader_type)
        self.dtype = np.dtype(dtype)
        self.low = low
        self.high = high

    def __iter__(self):
        return iter(self.utt2shape)

    def __len__(self):
        return len(self.utt2shape)

    def __getitem__(self, item) -> np.ndarray:
        shape = self.utt2shape[item]
        return np.random.randint(self.low, self.high, size=shape, dtype=self.dtype)
