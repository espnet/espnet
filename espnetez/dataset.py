"""ESPnetEZDataset module.

This module provides :class:`~ESPnetEZDataset`, a lightweight dataset wrapper for
ESPnet that simplifies access to data items and their associated metadata.
It is designed to be used in training pipelines where a simple mapping
from a unique identifier to a dictionary of feature values is required.

The wrapper accepts any sequence (list, tuple, or any ``Sized`` iterable)
containing dataset entries.  A ``data_info`` dictionary maps attribute names
to callables that extract the corresponding value from a single dataset
entry.  The class inherits from :class:`espnet2.train.dataset.AbsDataset`,
providing the standard ``__getitem__`` and ``__len__`` interfaces as well
as helper methods for introspection.

Typical use case
----------------
>>> dataset = [
...     ("audio1.wav", "transcription1"),
...     ("audio2.wav", "transcription2")
... ]
>>> data_info = {
...     "audio": lambda x: x[0],
...     "transcription": lambda x: x[1]
... }
>>> ez_dataset = ESPnetEZDataset(dataset, data_info)
>>> ez_dataset.has_name("audio")
True
>>> ez_dataset.names()
('audio', 'transcription')
>>> ez_dataset[0]
('0', {'audio': 'audio1.wav', 'transcription': 'transcription1'})
>>> len(ez_dataset)
2

Design goals
------------
* **Simplicity** - The API mirrors the familiar ``list`` interface while
  providing an attribute-based view of each item.
* **Flexibility** - ``data_info`` can contain any callables; they are
  evaluated lazily on item access.
* **Compatibility** - The class is compatible with ESPnet training loops
  that expect an ``AbsDataset`` implementation.

Attributes
----------
dataset : ``Sequence`` of entries
    The raw dataset from which items are retrieved.
data_info : ``dict[str, Callable]``
    Mapping from feature names to callables that compute the feature
    from a single dataset entry.

Methods
-------
has_name(name)
    Return ``True`` if *name* is a key in ``data_info``.
names()
    Return a tuple of all feature names defined in ``data_info``.
__getitem__(uid)
    Retrieve the entry with the given unique identifier.  ``uid`` may be
    a string or an integer; it is cast to ``int`` internally.
__len__()
    Return the number of entries in the dataset.

The module deliberately avoids external dependencies beyond the standard
library and ESPnet's :class:`AbsDataset`.  It is suitable for quick prototyping
or for embedding in larger ESPnet training scripts.
"""

from typing import Dict, Tuple, Union

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    """A dataset class for handling ESPnet data with easy access to data information.

    This class extends the AbsDataset class and provides functionalities to
    manage a dataset and its associated metadata. It allows users to retrieve
    dataset items using unique identifiers and check for available names in
    the dataset.

    Attributes:
        dataset (Union[list, Tuple]): The dataset containing the actual data entries.
        data_info (Dict[str, callable]): A dictionary mapping attribute names to
            functions that extract those attributes from the dataset.

    Args:
        dataset (Union[list, Tuple]): The dataset from which data will be extracted.
        data_info (Dict[str, callable]): A dictionary where keys are attribute names
            and values are functions that process the dataset entries.

    Methods:
        has_name(name): Checks if the given name exists in the data_info dictionary.
        names() -> Tuple[str, ...]: Returns a tuple of all names in the data_info.
        __getitem__(uid: Union[str, int]) -> Tuple[str, Dict]: Retrieves the data
            entry corresponding to the provided unique identifier.
        __len__() -> int: Returns the total number of entries in the dataset.

    Examples:
        >>> dataset = [
            ("audio1.wav", "transcription1"),
            ("audio2.wav", "transcription2")
        ]
        >>> data_info = {
        ...     "audio": lambda x: x[0],
        ...     "transcription": lambda x: x[1]
        ... }
        >>> ez_dataset = ESPnetEZDataset(dataset, data_info)
        >>> ez_dataset.has_name("audio")
        True
        >>> ez_dataset.names()
        ('audio', 'transcription')
        >>> ez_dataset[0]
        ('0', {'audio': 'audio1.wav', 'transcription': 'transcription1'})
        >>> len(ez_dataset)
        2

    Note:
        The dataset and data_info must be provided in a compatible format to ensure
        proper functionality of the methods.
    """

    def __init__(self, dataset, data_info):
        """Initialize the object with a dataset and its corresponding data.

        Args:
            dataset: The dataset to be stored in the instance. It can be any object
                     that represents the data to be processed or analyzed.
            data_info: Additional information about the dataset, such as metadata or
                       configuration parameters. This should be a dictionary or
                       any other mapping structure.
        """
        self.dataset = dataset
        self.data_info = data_info

    def has_name(self, name) -> bool:
        """Check if the specified name exists in the dataset's data information.

        This method searches the `data_info` attribute of the dataset to determine
        if the given `name` is present as a key. It is useful for validating
        whether certain attributes or features are available in the dataset.

        Args:
            name (str): The name to search for in the dataset's data information.

        Returns:
            bool: True if the name exists in the data information; False otherwise.

        Examples:
            >>> dataset = ESPnetEZDataset(dataset=[...],
                data_info={'feature1': ..., 'feature2': ...})
            >>> dataset.has_name('feature1')
            True
            >>> dataset.has_name('feature3')
            False

        Note:
            The method performs a simple membership check using the `in` operator,
            which is efficient for dictionaries.
        """
        return name in self.data_info

    def names(self) -> Tuple[str, ...]:
        """Handle data retrieval and management.

        This class extends the abstract dataset class to provide functionalities
        specific to the ESPnet framework. It manages a dataset and its associated
        metadata, allowing for efficient data access and manipulation.

        Attributes:
            dataset (Union[list, tuple]): The underlying dataset that contains the data.
            data_info (Dict[str, callable]): A dictionary mapping names to functions
                that process each data entry in the dataset.

        Args:
            dataset (Union[list, tuple]): The dataset to be wrapped.
            data_info (Dict[str, callable]): A dictionary where keys are the names of
                the data attributes and values are functions that extract or transform
                the data from the dataset.

        Methods:
            has_name(name: str) -> bool:
                Checks if a given name exists in the data_info.

            names() -> Tuple[str, ...]:
                Returns a tuple of all the names available in the data_info.

            __getitem__(uid: Union[str, int]) -> Tuple[str, Dict]:
                Retrieves the data entry corresponding to the provided identifier.

            __len__() -> int:
                Returns the number of entries in the dataset.

        Examples:
            >>> dataset = ESPnetEZDataset(dataset=[...],
                data_info={'feature': lambda x: x.feature, 'label': lambda x: x.label})
            >>> dataset.has_name('feature')
            True
            >>> dataset.names()
            ('feature', 'label')
            >>> entry = dataset[0]
            >>> print(entry)
            ('0', {'feature': ..., 'label': ...})
            >>> len(dataset)
            100

        Note:
            The functions provided in the data_info should be callable and should
            accept a single argument corresponding to an entry from the dataset.
        """
        return tuple(self.data_info.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict]:
        """Retrieve a data sample by identifier.

        This method is called when a user indexes the dataset object with ``[]``.
        The identifier ``uid`` can be either a string or an integer.  It is
        interpreted as the index of the underlying ``self.dataset`` after
        converting it to an integer.  The method returns a tuple containing
        the original identifier (converted to a string) and a dictionary with
        processed data values.

        The dictionary is constructed by iterating over ``self.data_info`` - a
        mapping from field names to callables.  Each callable is invoked with
        the raw dataset entry ``self.dataset[idx]`` to produce the processed
        value for that field.

        Args:
            uid: The identifier for the desired data sample.  It may be an
                integer index or a string that can be cast to an integer.
                The string is returned unchanged in the result tuple.

        Returns:
            Tuple[str, Dict]:
                * A string representation of ``uid``.
                * A dictionary where each key is from ``self.data_info`` and
                each value is the result of applying the corresponding
                callable to the indexed dataset entry.

        Raises:
            ValueError: If ``uid`` cannot be converted to an integer.
            IndexError: If the resulting index is out of bounds for
                ``self.dataset``.
        """
        idx = int(uid)
        return (
            str(uid),
            {k: v(self.dataset[idx]) for k, v in self.data_info.items()},
        )

    def __len__(self) -> int:
        """Return the number of items in the underlying dataset.

        This method implements the ``__len__`` protocol, allowing instances of the
        class to be used with the built-in ``len()`` function. It simply forwards
        the length calculation to the ``dataset`` attribute, which must support
        the ``len()`` operation.

        Returns:
            int: The number of elements contained in ``self.dataset``.
        """
        return len(self.dataset)
