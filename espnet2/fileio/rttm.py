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


def load_rttm_text(path: Union[Path, str]) -> (Dict[str, List[Tuple[str, float, float]]], List[str]):
    """ Read a RTTM file

        Note: only support speaker information now
    """

    assert check_argument_types()
    spk_index = 0
    data = {}
    spk_dict = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            assert(len(sps) == 9 and path)
            label_type, utt_id, channel, start, duration, _, _, spk_id, _ = sps
            
            # Only support speaker label now
            assert label_type == "SPEAKER"

            if spk_id in spk_dict.keys():
                spk_dict[spk_id] = spk_index
                spk_index += 1
            data[utt_id] = data.get(utt_id, []) + [(spk_id, float(start), float(start) + float(duration))]
    
    return data, spk_dict


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
        hop_length: int = 128,
        sample_rate: Union[int, str] = 16000,
    ):
        assert check_argument_types()
        super().__init__()

        self.fname = fname
        self.hop_length = hop_length
        if isinstance(sample_rate, str):
            self.sample_rate = humanfriendly.parse_size(sample_rate)
        self.data, self.spk_dict = load_rttm_text(path=fname)
        self.total_spk_num = len(self.spk_dict)
    
    def _get_duration_spk(self, spk_event: List[Tuple[str, float, float]]) -> Tuple(float, Set[str]):
        return max(map(lambda x: x[2], spk_event))

    def __getitem__(self, key):
        spk_event = self.data[key]
        max_duration = self._get_duration_spk(spk_event)
        size = np.rint(max_duration * self.sample_rate / self.hop_length).astype(int) + 1
        spk_label = np.zeros((size, self.total_spk_num))
        for spk_id, start, end in spk_event:
            start_frame = np.rint(start * self.sample_rate / self.hop_length).astype(int)
            end_frame = np.rint(end * self.sample_rate / self.hop_length).astype(int)
            spk_label[self.spk_dict[spk_id]][start_frame: end_frame + 1] = 1
        return spk_label

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
    
