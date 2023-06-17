import collections.abc
import logging
from mmap import mmap
from pathlib import Path
from random import randint
from typing import Dict, List, Union

from typeguard import check_argument_types


def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    assert check_argument_types()

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


def load_num_sequence_text(
    path: Union[Path, str], loader_type: str = "csv_int"
) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number

    Examples:
        key1 1 2 3
        key2 34 5 6

        >>> d = load_num_sequence_text('text')
        >>> np.testing.assert_array_equal(d["key1"], np.array([1, 2, 3]))
    """
    assert check_argument_types()
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
    d = read_2column_text(path)

    # Using for-loop instead of dict-comprehension for debuggability
    retval = {}
    for k, v in d.items():
        try:
            retval[k] = [dtype(i) for i in v.split(delimiter)]
        except TypeError:
            logging.error(f'Error happened with path="{path}", id="{k}", value="{v}"')
            raise
    return retval


def read_label(path: Union[Path, str]) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number

    Examples:
        key1 start_time_1 end_time_1 phone_1 start_time_2 end_time_2 phone_2 ....\n
        key2 start_time_1 end_time_1 phone_1 \n

        >>> d = load_num_sequence_text('label')
        >>> np.testing.assert_array_equal(d["key1"], [0.1, 0.2, "啊"]))
    """
    assert check_argument_types()
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
    """Reader class for random access to text.

    Simple text reader for non-pair text data (for unsupervised ASR)
        Instead of loading the whole text into memory (often large for UASR),
        the reader consumes text which stores in byte-offset of each text file
        and randomly selected unpaired text from it for training using mmap.

    Examples:
        text
            text1line
            text2line
            text3line
        scp
            11
            00000000000000000010
            00000000110000000020
            00000000210000000030
        scp explanation
            (number of digits per int value)
            (text start at bytes 0 and end at bytes 10 (including "\n"))
            (text start at bytes 11 and end at bytes 20 (including "\n"))
            (text start at bytes 21 and end at bytes 30 (including "\n"))
    """

    def __init__(
        self,
        text_and_scp: str,
    ):
        assert check_argument_types()
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
        return None
