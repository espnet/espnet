import warnings
from pathlib import Path
from typing import Union

from typeguard import typechecked


class DatadirWriter:
    """
    Writer class to create a Kaldi-like data directory.

    This class facilitates the creation of a structured data directory
    similar to those used in Kaldi. It allows for writing key-value pairs
    representing utterance IDs and their corresponding audio file paths,
    as well as creating subdirectories for organizing the data.

    Examples:
        >>> with DatadirWriter("output") as writer:
        ...     # output/sub.txt is created here
        ...     subwriter = writer["sub.txt"]
        ...     # Write "uttidA some/where/a.wav"
        ...     subwriter["uttidA"] = "some/where/a.wav"
        ...     subwriter["uttidB"] = "some/where/b.wav"

    Attributes:
        path (Path): The path to the data directory being created.
        children (dict): A dictionary holding references to child
            DatadirWriter instances.
        fd (TextIOWrapper or None): File descriptor for writing to a
            file, or None if writing to a directory.
        has_children (bool): Flag indicating if there are child
            DatadirWriter instances.
        keys (set): A set of keys that have been written to the
            current data directory or file.

    Args:
        p (Union[Path, str]): The path where the data directory
            should be created.

    Raises:
        RuntimeError: If attempting to write to a file when a
            subdirectory exists or vice versa.

    Note:
        The `__enter__` and `__exit__` methods allow for the use
        of this class in a context manager, ensuring that resources
        are properly managed.

    Todo:
        - Implement functionality to validate file paths before
          writing.
    """

    @typechecked
    def __init__(self, p: Union[Path, str]):
        self.path = Path(p)
        self.chilidren = {}
        self.fd = None
        self.has_children = False
        self.keys = set()

    def __enter__(self):
        return self

    @typechecked
    def __getitem__(self, key: str) -> "DatadirWriter":
        if self.fd is not None:
            raise RuntimeError("This writer points out a file")

        if key not in self.chilidren:
            w = DatadirWriter((self.path / key))
            self.chilidren[key] = w
            self.has_children = True

        retval = self.chilidren[key]
        return retval

    @typechecked
    def __setitem__(self, key: str, value: str):
        if self.has_children:
            raise RuntimeError("This writer points out a directory")
        if key in self.keys:
            warnings.warn(f"Duplicated: {key}")

        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = self.path.open("w", encoding="utf-8")

        self.keys.add(key)
        self.fd.write(f"{key} {value}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Close the data writer and all associated child writers.

        This method is responsible for properly closing the current writer
        instance. If there are any child writers, it will recursively close
        each of them and will also check for key mismatches between siblings,
        issuing warnings if discrepancies are found. If the current writer
        has an open file descriptor, it will be closed as well.

        Raises:
            RuntimeError: If there is an issue while closing the file
            descriptor or child writers.

        Examples:
            >>> with DatadirWriter("output") as writer:
            ...     subwriter = writer["sub.txt"]
            ...     subwriter["uttidA"] = "some/where/a.wav"
            ...     # When exiting the context, close is called automatically.

        Note:
            This method is automatically invoked when exiting the context
            manager. It is recommended not to call this method directly
            unless you are managing the writer lifecycle manually.
        """
        if self.has_children:
            prev_child = None
            for child in self.chilidren.values():
                child.close()
                if prev_child is not None and prev_child.keys != child.keys:
                    warnings.warn(
                        f"Ids are mismatching between "
                        f"{prev_child.path} and {child.path}"
                    )
                prev_child = child

        elif self.fd is not None:
            self.fd.close()
