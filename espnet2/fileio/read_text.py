import collections.abc
import logging
from mmap import mmap
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Tuple, Union

from typeguard import typechecked


@typechecked
def read_2columns_text(path: Union[Path, str]) -> Dict[str, str]:
    """
        Read a text file having 2 columns as dict object.

    This function reads a text file where each line contains two columns,
    separated by whitespace. The first column is treated as the key and the
    second column as the value. It returns a dictionary mapping each key
    to its corresponding value. If duplicate keys are found, a RuntimeError
    is raised.

    Args:
        path (Union[Path, str]): The path to the text file to be read.

    Returns:
        Dict[str, str]: A dictionary containing key-value pairs from the
        text file.

    Raises:
        RuntimeError: If a duplicate key is found in the text file.

    Examples:
        Given a file named `wav.scp` with the following content:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        The function can be called as follows:
            >>> read_2columns_text('wav.scp')
            {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    Note:
        The function expects the text file to be encoded in UTF-8.
    """

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps

            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data


@typechecked
def read_multi_columns_text(
    path: Union[Path, str], return_unsplit: bool = False
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, str]]]:
    """
    Read a text file having 2 or more columns as dict object.

    This function reads a text file where each line contains a key followed
    by one or more values separated by whitespace. The function returns a
    dictionary where the keys are the first column values, and the values
    are lists of strings representing the remaining columns. Optionally,
    it can also return the unsplit raw values.

    Args:
        path (Union[Path, str]): The path to the text file to be read.
        return_unsplit (bool): If True, return a second dictionary with
            unsplit values (default is False).

    Returns:
        Tuple[Dict[str, List[str]], Optional[Dict[str, str]]]: A tuple
        containing:
            - A dictionary where keys are the first column values and
              values are lists of strings for the remaining columns.
            - An optional dictionary with the unsplit raw values if
              return_unsplit is True; otherwise, None.

    Raises:
        RuntimeError: If a key is duplicated in the input file.

    Examples:
        Given a file 'wav.scp' with the following content:
            key1 /some/path/a1.wav /some/path/a2.wav
            key2 /some/path/b1.wav /some/path/b2.wav /some/path/b3.wav
            key3 /some/path/c1.wav

        >>> read_multi_columns_text('wav.scp')
        {'key1': ['/some/path/a1.wav', '/some/path/a2.wav'],
         'key2': ['/some/path/b1.wav', '/some/path/b2.wav',
                  '/some/path/b3.wav'],
         'key3': ['/some/path/c1.wav']}

        If return_unsplit is True:
        >>> read_multi_columns_text('wav.scp', return_unsplit=True)
        ({'key1': ['/some/path/a1.wav', '/some/path/a2.wav'],
          'key2': ['/some/path/b1.wav', '/some/path/b2.wav',
                   '/some/path/b3.wav'],
          'key3': ['/some/path/c1.wav']},
         {'key1': '/some/path/a1.wav /some/path/a2.wav',
          'key2': '/some/path/b1.wav /some/path/b2.wav /some/path/b3.wav',
          'key3': '/some/path/c1.wav'})
    """

    data = {}

    if return_unsplit:
        unsplit_data = {}
    else:
        unsplit_data = None

    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps

            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")

            data[k] = v.split() if v != "" else [""]
            if return_unsplit:
                unsplit_data[k] = v

    return data, unsplit_data


@typechecked
def load_num_sequence_text(
    path: Union[Path, str], loader_type: str = "csv_int"
) -> Dict[str, List[Union[float, int]]]:
    """
        Load a text file indicating sequences of numbers.

    This function reads a text file where each line contains a key followed by a
    sequence of numbers. The numbers can be either integers or floats depending on
    the specified loader type. The function returns a dictionary where the keys are
    the first elements of each line and the values are lists of numbers parsed from
    the corresponding lines.

    Args:
        path (Union[Path, str]): The path to the text file to be read.
        loader_type (str): The format of the numbers in the file. Supported values
            are:
            - "text_int": Space-separated integers
            - "text_float": Space-separated floats
            - "csv_int": Comma-separated integers
            - "csv_float": Comma-separated floats

    Returns:
        Dict[str, List[Union[float, int]]]: A dictionary mapping keys to lists of
        numbers parsed from the text file.

    Raises:
        ValueError: If an unsupported loader_type is specified.

    Examples:
        Assuming the content of 'text' file is:
            key1 1 2 3
            key2 34 5 6

        >>> d = load_num_sequence_text('text')
        >>> print(d)
        {'key1': [1, 2, 3], 'key2': [34, 5, 6]}

        For floating point numbers:
        Assuming the content of 'text_float' file is:
            key1 1.0 2.5 3.3
            key2 34.1 5.6 6.2

        >>> d = load_num_sequence_text('text_float', loader_type='text_float')
        >>> print(d)
        {'key1': [1.0, 2.5, 3.3], 'key2': [34.1, 5.6, 6.2]}
    """
    if loader_type == "text_int":
        delimiter = " "
        dtype = int
    elif loader_type == "text_float":
        delimiter = " "
        dtype = float
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = int
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = float
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")

    # path looks like:
    #   utta 1,0
    #   uttb 3,4,5
    # -> return {'utta': np.ndarray([1, 0]),
    #            'uttb': np.ndarray([3, 4, 5])}
    d = read_2columns_text(path)

    # Using for-loop instead of dict-comprehension for debuggability
    retval = {}
    for k, v in d.items():
        try:
            retval[k] = [dtype(i) for i in v.split(delimiter)]
        except TypeError:
            logging.error(f'Error happened with path="{path}", id="{k}", value="{v}"')
            raise
    return retval


@typechecked
def read_label(path: Union[Path, str]) -> Dict[str, List[List[Union[str, float, int]]]]:
    """
        Read a text file indicating sequences of numbers or labels.

    This function reads a text file where each line consists of a key followed
    by a sequence of associated values. The values are expected to be in the
    format of start time, end time, and label for each segment. The output
    is a dictionary where each key maps to a list of lists containing the
    parsed information.

    Attributes:
        path (Union[Path, str]): The file path of the text file to be read.

    Args:
        path: A string or Path object representing the path to the text file.

    Returns:
        Dict[str, List[List[Union[str, float, int]]]]:
            A dictionary where each key corresponds to a sequence of
            labels and associated timing information, with the values
            being lists of lists that contain the start time, end time,
            and label.

    Examples:
        Given a file 'label.txt' with the following content:
            key1 0.1 0.2 "啊" 0.3 0.4 "喔"
            key2 0.5 0.6 "哦"

        The function can be used as follows:
            >>> d = read_label('label.txt')
            >>> print(d)
            {'key1': [['0.1', '0.2', '啊'], ['0.3', '0.4', '喔']],
             'key2': [['0.5', '0.6', '哦']]}

    Note:
        The input file must be formatted correctly, with each line
        starting with a unique key followed by the respective data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IndexError: If a line in the file does not contain enough values
                    to be parsed correctly.
    """
    label = open(path, "r", encoding="utf-8")

    retval = {}
    for label_line in label.readlines():
        line = label_line.strip().split()
        key = line[0]
        phn_info = line[1:]
        temp_info = []
        for i in range(len(phn_info) // 3):
            temp_info.append(
                [phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2]]
            )
        retval[key] = temp_info
    return retval


class RandomTextReader(collections.abc.Mapping):
    """
    Reader class for random access to text.

    This class provides a simple text reader for non-pair text data,
    particularly useful for unsupervised automatic speech recognition (ASR).
    Instead of loading the entire text into memory (which can be large for
    unsupervised ASR), the reader uses memory mapping (mmap) to access text
    stored in byte-offsets within each text file. This allows for random
    selection of unpaired text for training.

    Attributes:
        text_mm (mmap): Memory-mapped object for the text file.
        scp_mm (mmap): Memory-mapped object for the SCP file.
        first_line_offset (int): The byte offset of the first line in the SCP file.
        max_num_digits (int): The maximum number of digits per line in the SCP file.
        stride (int): The total number of bytes per line in the SCP file.
        num_lines (int): The total number of lines in the text file.

    Args:
        text_and_scp (str): A string containing the paths to the text file and
            the SCP file, separated by a hyphen (e.g., "text.txt-scp.txt").

    Examples:
        Suppose you have a text file with the following content:
            text1line
            text2line
            text3line

        And an SCP file that looks like this:
            11
            00000000000000000010
            00000000110000000020
            00000000210000000030

        You can create an instance of the RandomTextReader like this:
        >>> reader = RandomTextReader("text.txt-scp.txt")
        Then, you can access random lines from the text:
        >>> random_line = reader[0]  # Access a random line

    Note:
        The SCP file format must follow the specified structure for the reader
        to function correctly.

    Raises:
        AssertionError: If the SCP file does not contain valid data or if
        the number of bytes is not consistent.
    """

    @typechecked
    def __init__(
        self,
        text_and_scp: str,
    ):
        super().__init__()

        text, text_scp = text_and_scp.split("-")

        text_f = Path(text).open("r+b")
        scp_f = Path(text_scp).open("r+b")

        self.text_mm = mmap(text_f.fileno(), 0)
        self.scp_mm = mmap(scp_f.fileno(), 0)

        max_num_digits_line = self.scp_mm.readline()
        max_num_digits = int(max_num_digits_line)
        assert max_num_digits > 0

        self.first_line_offset = len(max_num_digits_line)
        self.max_num_digits = max_num_digits
        self.stride = 2 * max_num_digits + 1

        num_text_bites = len(self.scp_mm) - len(max_num_digits_line)
        assert num_text_bites % self.stride == 0
        num_lines = num_text_bites // self.stride
        self.num_lines = num_lines

    def __getitem__(self, key):
        # choose random line from scp
        # the first line defines the max number of digits
        random_line_number = randint(0, self.num_lines - 1)

        # get the number of bytes of corresponding line in text
        scp_start_bytes = self.first_line_offset
        scp_start_bytes += random_line_number * self.stride
        scp_end_bytes = scp_start_bytes + self.stride - 1

        text_start_bytes = int(
            self.scp_mm[scp_start_bytes : scp_start_bytes + self.max_num_digits]
        )
        text_end_bytes = int(
            self.scp_mm[scp_start_bytes + self.max_num_digits : scp_end_bytes]
        )

        # retrieve text line
        text = self.text_mm[text_start_bytes:text_end_bytes].decode("utf-8")
        return text

    def __contains__(self, item):
        return True

    def __len__(self):
        return self.num_lines

    def __iter__(self):
        return None

    def keys(self):
        """
            Reader class for random access to text.

        This class provides a simple text reader for non-paired text data, which is
        useful for unsupervised automatic speech recognition (ASR). Instead of loading
        the entire text into memory (which can be large for UASR), the reader utilizes
        memory-mapped files to efficiently access text stored in byte offsets. This
        allows for random selection of unpaired text for training.

        Attributes:
            text_mm (mmap): Memory-mapped object for the text file.
            scp_mm (mmap): Memory-mapped object for the SCP file.
            first_line_offset (int): Offset of the first line in the SCP file.
            max_num_digits (int): Maximum number of digits in the SCP file.
            stride (int): The number of bytes for each line in the SCP file.
            num_lines (int): The total number of lines in the text file.

        Args:
            text_and_scp (str): A string containing the paths to the text file and
                the SCP file, separated by a hyphen (e.g., 'text.txt-scp.txt').

        Examples:
            Given a text file with lines:
                text1line
                text2line
                text3line

            And a corresponding SCP file:
                11
                00000000000000000010
                00000000110000000020
                00000000210000000030

            You can create a RandomTextReader instance and access lines as follows:

            >>> reader = RandomTextReader('text.txt-scp.txt')
            >>> print(reader[0])  # Outputs one of the text lines randomly
            >>> print(len(reader))  # Outputs 3, the number of text lines

        Note:
            The SCP file format requires that the number of bytes specified for each
            line in the SCP file corresponds correctly to the lines in the text file.

        Raises:
            AssertionError: If the maximum number of digits read from the SCP file is
                less than or equal to zero or if the number of text bytes is not
                divisible by the stride.
        """
        return None
