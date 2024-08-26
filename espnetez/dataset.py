from typing import Dict, Tuple, Union

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    """
    A dataset class for handling ESPnet data with easy access to data information.

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
        self.dataset = dataset
        self.data_info = data_info

    def has_name(self, name) -> bool:
        """
        Check if the specified name exists in the dataset's data information.

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
        """
            A dataset class for ESPnet that handles data retrieval and management.

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
        idx = int(uid)
        return (
            str(uid),
            {k: v(self.dataset[idx]) for k, v in self.data_info.items()},
        )

    def __len__(self) -> int:
        return len(self.dataset)
