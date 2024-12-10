import collections.abc
from pathlib import Path
from typing import List, Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text


class VADScpReader(collections.abc.Mapping):
    """
    Reader class for 'vad.scp'.

    This class provides functionality to read a 'vad.scp' file, which focuses
    on utterance-level voice activity detection (VAD) segments. Unlike the
    `segments` file, which encompasses entire sessions, the `vad.scp` file is
    designed to guide silence trimming for UASR (Unsupervised Automatic
    Speech Recognition).

    Attributes:
        fname (str): The file name of the 'vad.scp' file to read.
        dtype (numpy.dtype): The data type for the VAD segments, defaulting to
            np.float32.
        data (dict): A dictionary mapping keys to VAD segments.

    Args:
        fname (str): Path to the 'vad.scp' file.
        dtype (numpy.dtype, optional): Data type for VAD segments.

    Returns:
        dict: A mapping of keys to their respective VAD segments.

    Examples:
        >>> reader = VADScpReader('vad.scp')
        >>> array = reader['key1']
        # array will contain the VAD segments for 'key1' as a list of tuples.

    Raises:
        KeyError: If the key is not found in the 'vad.scp' file.
        ValueError: If the VAD segment format is invalid.
    """

    @typechecked
    def __init__(
        self,
        fname,
        dtype=np.float32,
    ):
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)

    def __getitem__(self, key):
        vads = self.data[key]
        vads = vads.split(" ")
        vad_info = []
        for vad in vads:
            start, end = vad.split(":")
            vad_info.append((float(start), float(end)))
        return vad_info

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for 'vad.scp'.

        This class reads a VAD (Voice Activity Detection) script file, which is
        used to guide the silence trimming for UASR (Unsupervised Automatic
        Speech Recognition). Unlike the `segments`, which focus on whole sessions,
        the `vad.scp` file focuses on utterance-level information.

        Attributes:
            fname (str): The filename of the VAD script file.
            dtype (numpy.dtype): The data type for the VAD values.
            data (dict): A dictionary containing the VAD data parsed from the file.

        Args:
            fname (str): Path to the 'vad.scp' file to read.
            dtype (numpy.dtype, optional): The data type for VAD values.
                Defaults to np.float32.

        Returns:
            List[Tuple[float, float]]: A list of tuples representing the VAD
            intervals for a given key.

        Examples:
            key1 0:1.2000
            key2 3.0000:4.5000 7.0000:9:0000
            ...

            >>> reader = VADScpReader('vad.scp')
            >>> array = reader['key1']

        Raises:
            KeyError: If the specified key does not exist in the VAD data.

        Note:
            The `vad.scp` file format expects each line to contain a key followed
            by one or more time intervals, separated by spaces. Each interval is
            represented as `start:end`.
        """
        return self.data.keys()


class VADScpWriter:
    """
    Writer class for 'vad.scp'.

    This class provides functionality to write Voice Activity
    Detection (VAD) data to a 'vad.scp' file. The VAD data is
    represented as utterance-level segments, which can be used
    to guide silence trimming for tasks like Unsupervised
    Automatic Speech Recognition (UASR).

    Attributes:
        scpfile (Path): The path to the 'vad.scp' file.
        dtype: Data type for the VAD values (optional).
        data (dict): A dictionary to store VAD entries.

    Args:
        scpfile (Union[Path, str]): Path to the output 'vad.scp' file.
        dtype: Data type for the VAD values (default is None).

    Examples:
        key1 0:1.2000
        key2 3.0000:4.5000 7.0000:9.0000
        ...

        >>> writer = VADScpWriter('./data/vad.scp')
        >>> writer['aa'] = [(0.0, 1.2), (2.0, 3.0)]
        >>> writer['bb'] = [(3.0, 4.5)]

    Raises:
        AssertionError: If a duplicate key is found or if the
                        provided value is not a list of tuples
                        with exactly two elements each.

    Note:
        Ensure that the keys used are unique within the VAD
        entries to avoid overwriting data.

    Todo:
        Add support for appending to an existing file without
        overwriting previous entries.
    """

    @typechecked
    def __init__(
        self,
        scpfile: Union[Path, str],
        dtype=None,
    ):
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value):
        assert (
            key not in self.data.keys()
        ), "found duplicate key (key: {}) in your vad values".format(key)
        assert isinstance(value, List), type(value)

        output_str = []
        for v in value:
            assert (
                len(v) == 2
            ), "each vad tuple should contains exact the start time and end time"
            output_str.append("{.4f}:{}".format(v[0], v[1]))
        output_str = " ".join(output_str)

        self.fscp.write(f"{key} {output_str}\n")

        # Store the file path
        self.data[key] = str(output_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Writer class for 'vad.scp'.

            This class is responsible for writing voice activity detection (VAD) data
            to a file in the 'vad.scp' format. The format consists of keys associated
            with time intervals, indicating when speech is detected. This is useful
            for applications such as automatic speech recognition (ASR) to manage
            silence trimming.

            The output format is as follows:
                key1 0:1.2000
                key2 3.0000:4.5000 7.0000:9.0000
                ...

            Examples:
                >>> writer = VADScpWriter('./data/vad.scp')
                >>> writer['aa'] = [(0.0, 1.2), (2.0, 3.5)]
                >>> writer['bb'] = [(1.0, 2.0)]

            Attributes:
                scpfile (Path): The path to the output 'vad.scp' file.
                dtype (optional): Data type for the VAD intervals.
                data (dict): A dictionary to store the VAD data associated with keys.

            Args:
                scpfile (Union[Path, str]): The path to the 'vad.scp' file to be created.
                dtype (optional): Data type for the VAD intervals. Defaults to None.

            Raises:
                AssertionError: If a duplicate key is found or if the value is not a list
                                of tuples.

            Note:
                This class should be used in a context manager to ensure that the file
                is properly closed after writing.

            Todo:
                - Implement additional validation for the input VAD tuples.
        """
        self.fscp.close()
