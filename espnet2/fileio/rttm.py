import collections.abc
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from typeguard import typechecked


@typechecked
def load_rttm_text(path: Union[Path, str]) -> Dict[str, List[Tuple[str, float, float]]]:
    """
        Read a RTTM (Rich Transcription Time Marked) file and extract speaker
    information.

    This function reads a RTTM file and organizes the speaker annotations
    into a structured dictionary. The dictionary maps utterance IDs to a
    list of tuples containing speaker IDs and their corresponding start
    and end times.

    Note: This function currently only supports speaker information.

    Args:
        path (Union[Path, str]): The file path to the RTTM file to be read.

    Returns:
        Dict[str, List[Tuple[str, float, float]]]: A dictionary where the keys
        are utterance IDs and the values are lists of tuples. Each tuple
        contains the speaker ID, start time, and end time for each speaker
        in the utterance.

    Raises:
        AssertionError: If the line in the RTTM file does not contain exactly
        9 fields or if the label type is not "SPEAKER" or "END".

    Examples:
        >>> rttm_data = load_rttm_text('path/to/rttm/file.rttm')
        >>> print(rttm_data)
        {
            'file1': [
                ('spk1', 0, 1023),
                ('spk2', 4000, 3023),
                ('spk1', 500, 4023)
            ]
        }
    """

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            assert len(sps) == 9, "{} does not have exactly 9 fields".format(path)
            label_type, utt_id, channel, start, end, _, _, spk_id, _ = sps

            # Only support speaker label now
            assert label_type in ["SPEAKER", "END"]

            spk_list, spk_event, max_duration = data.get(utt_id, ([], [], 0))
            if label_type == "END":
                data[utt_id] = (spk_list, spk_event, int(end))
                continue
            if spk_id not in spk_list:
                spk_list.append(spk_id)

            data[utt_id] = (
                spk_list,
                spk_event + [(spk_id, int(float(start)), int(float(end)))],
                max_duration,
            )

    return data


class RttmReader(collections.abc.Mapping):
    """
    Reader class for 'rttm.scp'.

    This class provides functionality to read RTTM (Rich Transcription Time
    Markup) files, specifically tailored for the ESPnet framework. The RTTM
    format supported by this class extends the standard format by using
    sample numbers instead of absolute time and includes an END label to
    represent the duration of a recording.

    The standard RTTM format can be found at:
    https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf

    Attributes:
        fname (str): The filename of the RTTM file to be read.
        data (Dict[str, List[Tuple[str, float, float]]]): Parsed RTTM data
            where keys are utterance IDs and values are tuples containing
            speaker list, speaker events, and maximum duration.

    Args:
        fname (str): The path to the RTTM file.

    Examples:
        >>> reader = RttmReader('rttm')
        >>> spk_label = reader["file1"]

        The RTTM file may contain lines such as:
        SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>
        SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>
        SPEAKER file1 3 500 4023 <NA> <NA> spk1 <NA>
        END     file1 <NA> 4023 <NA> <NA> <NA> <NA>

        This example shows how to instantiate the reader and access the
        speaker labels for a given file.

    Note:
        The reader currently supports only speaker information.
        Ensure that the RTTM file is formatted correctly to avoid
        assertion errors.

    Raises:
        AssertionError: If the RTTM line does not have exactly 9 fields or
            if the label type is not "SPEAKER" or "END".
    """

    @typechecked
    def __init__(
        self,
        fname: str,
    ):
        super().__init__()

        self.fname = fname
        self.data = load_rttm_text(path=fname)

    def __getitem__(self, key):
        spk_list, spk_event, max_duration = self.data[key]
        spk_label = np.zeros((max_duration, len(spk_list)))
        for spk_id, start, end in spk_event:
            spk_label[start : end + 1, spk_list.index(spk_id)] = 1
        return spk_label

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
                Read and parse RTTM (Rich Transcription Time Marked) files.

        This module provides functionality to read RTTM files and extract speaker
        information. The `load_rttm_text` function reads the file and returns a
        dictionary containing speaker events associated with each utterance.

        Note: This implementation currently only supports speaker information.

        Attributes:
            - RttmReader: A class for reading RTTM files.

        Args:
            path (Union[Path, str]): The file path to the RTTM file to be read.

        Returns:
            Dict[str, List[Tuple[str, float, float]]]: A dictionary where each key
            is an utterance ID and the value is a list of tuples containing speaker
            ID, start time, and end time.

        Raises:
            AssertionError: If the RTTM line does not contain exactly 9 fields or if
            the label type is not "SPEAKER" or "END".

        Examples:
            >>> data = load_rttm_text("path/to/rttm/file.rttm")
            >>> print(data)
            {'file1': (['spk1', 'spk2'], [(spk1, start1, end1), (spk2, start2, end2)], max_duration)}

        RttmReader class:
            Reader class for 'rttm.scp'.

            Examples:
                SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>
                SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>
                SPEAKER file1 3 500 4023 <NA> <NA> spk1 <NA>
                END     file1 <NA> 4023 <NA> <NA> <NA> <NA>

                This is an extended version of the standard RTTM format for espnet.
                The differences include:
                1. Use of sample number instead of absolute time.
                2. Inclusion of an END label to represent the duration of a recording.
                3. Replacement of duration (5th field) with end time.
                (For standard RTTM, see
                https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)

            Examples:
                >>> reader = RttmReader('path/to/rttm/file.rttm')
                >>> spk_label = reader["file1"]
        """
        return self.data.keys()
