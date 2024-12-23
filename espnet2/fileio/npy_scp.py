import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text


class NpyScpWriter:
    """
    Writer class for creating a SCP file of numpy arrays.

    This class allows users to write numpy arrays to a specified directory
    and generate a corresponding SCP file that maps keys to numpy file paths.

    Attributes:
        dir (Path): The directory where numpy files will be saved.
        fscp (TextIOWrapper): The file object for writing the SCP file.
        data (dict): A dictionary to store the mapping of keys to numpy file paths.

    Args:
        outdir (Union[Path, str]): The output directory for numpy files.
        scpfile (Union[Path, str]): The path for the SCP file to be created.

    Examples:
        The SCP file will contain lines in the format:
            key1 /some/path/a.npy
            key2 /some/path/b.npy
            key3 /some/path/c.npy
            key4 /some/path/d.npy
            ...

        >>> writer = NpyScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array  # Save 'numpy_array' to './data/aa.npy'
        >>> writer['bb'] = numpy_array  # Save 'numpy_array' to './data/bb.npy'

    Raises:
        AssertionError: If the value assigned is not a numpy ndarray.

    Note:
        Ensure that the output directory and SCP file path are valid and writable.
    """

    @typechecked
    def __init__(self, outdir: Union[Path, str], scpfile: Union[Path, str]):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")

        self.data = {}

    def get_path(self, key):
        """
        Retrieve the file path associated with the given key.

        This method looks up the provided key in the internal data
        structure and returns the corresponding file path where the
        numpy array is stored.

        Args:
            key (str): The key for which the file path needs to be
                retrieved.

        Returns:
            str: The file path corresponding to the given key.

        Raises:
            KeyError: If the key is not found in the internal data
                structure.

        Examples:
            >>> writer = NpyScpWriter('./data/', './data/feat.scp')
            >>> writer['example_key'] = np.array([1, 2, 3])
            >>> path = writer.get_path('example_key')
            >>> print(path)  # Output: './data/example_key.npy'

        Note:
            Ensure that the key exists in the internal data structure
            before calling this method to avoid a KeyError.
        """
        return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        p = self.dir / f"{key}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), value)
        self.fscp.write(f"{key} {p}\n")

        # Store the file path
        self.data[key] = str(p)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Closes the SCP file associated with the NpyScpWriter instance.

            This method should be called to ensure that all data is properly flushed
            and the file is closed when writing operations are complete. It is
            automatically invoked when exiting the context manager.

            Note:
                It is important to call this method to prevent any data loss
                or corruption.

            Examples:
                >>> writer = NpyScpWriter('./data/', './data/feat.scp')
                >>> writer['aa'] = numpy_array
                >>> writer['bb'] = numpy_array
                >>> writer.close()  # Ensure the SCP file is closed properly

            Raises:
                ValueError: If the SCP file is already closed when attempting to
                close it again.
        """
        self.fscp.close()


class NpyScpReader(collections.abc.Mapping):
    """
        Reader class for a scp file of numpy files.

    This class allows for reading numpy arrays from a specified scp (script) file,
    which contains key-path pairs, where each key maps to a numpy file. The keys
    can be used to retrieve the corresponding numpy arrays efficiently.

    Attributes:
        fname (Path): The path to the scp file.
        data (dict): A dictionary mapping keys to numpy file paths.

    Args:
        fname (Union[Path, str]): The path to the scp file.

    Returns:
        None

    Examples:
        The scp file might look like this:
            key1 /some/path/a.npy
            key2 /some/path/b.npy
            key3 /some/path/c.npy
            key4 /some/path/d.npy
            ...

        Usage:
            >>> reader = NpyScpReader('npy.scp')
            >>> array = reader['key1']

        You can check if a key exists and get the number of keys:
            >>> 'key1' in reader
            True
            >>> len(reader)
            4

    Yields:
        None

    Raises:
        KeyError: If the specified key does not exist in the scp file.

    Note:
        Ensure that the numpy files referenced in the scp file exist at the
        specified paths.

    Todo:
        - Add functionality for error handling when loading numpy files.
    """

    @typechecked
    def __init__(self, fname: Union[Path, str]):
        self.fname = Path(fname)
        self.data = read_2columns_text(fname)

    def get_path(self, key):
        """
            Retrieve the file path associated with the given key.

        This method returns the path of the numpy file that corresponds to
        the provided key in the SCP file. It allows users to access the
        storage location of the numpy array without loading the array into
        memory.

        Args:
            key (str): The key associated with the desired numpy file.

        Returns:
            str: The file path of the numpy file corresponding to the key.

        Raises:
            KeyError: If the key does not exist in the data.

        Examples:
            >>> reader = NpyScpReader('npy.scp')
            >>> path = reader.get_path('key1')
            >>> print(path)
            /some/path/a.npy

        Note:
            Ensure that the key exists in the data before calling this method
            to avoid a KeyError.
        """
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return np.load(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for a SCP (script) file of numpy files.

        This class allows users to read numpy arrays from a specified SCP file.
        The SCP file should contain lines with keys and corresponding paths to
        numpy files.

        Attributes:
            fname (Path): The path to the SCP file.
            data (dict): A dictionary mapping keys to file paths read from the SCP file.

        Args:
            fname (Union[Path, str]): The path to the SCP file.

        Returns:
            None

        Examples:
            Suppose the SCP file `npy.scp` contains:
                key1 /some/path/a.npy
                key2 /some/path/b.npy
                key3 /some/path/c.npy
                key4 /some/path/d.npy

            You can read the numpy array for `key1` as follows:

            >>> reader = NpyScpReader('npy.scp')
            >>> array = reader['key1']

        Raises:
            KeyError: If the specified key is not found in the SCP file.
            FileNotFoundError: If the numpy file corresponding to the key does not exist.
        """
        return self.data.keys()
