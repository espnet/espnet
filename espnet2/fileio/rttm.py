#!/usr/bin/env python3

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import collections.abc
import humanfriendly
from pathlib import Path
from typing import Dict
from typing import List
from typing import Mapping
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import re
from pathlib import Path
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


def load_rttm_text(
    path: Union[Path, str]
) -> (Dict[str, List[Tuple[str, float, float]]], List[str]):
    """Read a RTTM file

    Note: only support speaker information now
    """

    assert check_argument_types()
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            assert len(sps) == 9, "{} does not have exactly 9 fields".format(path)
            label_type, utt_id, channel, start, duration, _, _, spk_id, _ = sps

            # Only support speaker label now
            assert label_type == "SPEAKER"

            spk_list, spk_event = data.get(utt_id, ([], []))
            if spk_id not in spk_list:
                spk_list.append(spk_id)
            data[utt_id] = spk_list, spk_event + [
                (spk_id, float(start), float(start) + float(duration))
            ]

    return data


class RttmReader(collections.abc.Mapping):
    """Reader class for 'rttm.scp'.

    Examples:
        SPEAKER file1 1 0.00 1.23 <NA> <NA> spk1 <NA>
        SPEAKER file1 2 4.00 3.23 <NA> <NA> spk2 <NA>
        SPEAKER file1 3 5.00 4.23 <NA> <NA> spk1 <NA>
        (see https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)
        ...

        Note: only support speaker information now

        >>> reader = RttmReader('rttm')
        >>> spk_label = reader["file1"]

    """

    def __init__(
        self,
        fname: str,
        sample_rate: Union[int, str] = 16000,
    ):
        assert check_argument_types()
        super().__init__()

        self.fname = fname
        if isinstance(sample_rate, str):
            self.sample_rate = humanfriendly.parse_size(sample_rate)
        else:
            self.sample_rate = sample_rate
        self.data = load_rttm_text(path=fname)

    def _get_duration_spk(
        self, spk_event: List[Tuple[str, float, float]]
    ) -> Tuple[float, Set[str]]:
        return max(map(lambda x: x[2], spk_event))

    def __getitem__(self, key):
        spk_list, spk_event = self.data[key]
        max_duration = self._get_duration_spk(spk_event)
        size = np.rint(max_duration * self.sample_rate).astype(int) + 1
        spk_label = np.zeros((size, len(spk_list)))
        for spk_id, start, end in spk_event:
            start_sample = np.rint(start * self.sample_rate).astype(int)
            end_sample = np.rint(end * self.sample_rate).astype(int)
            spk_label[spk_list.index(spk_id)][start_sample : end_sample + 1] = 1
        return spk_label

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
