import collections.abc
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from typeguard import check_argument_types


def load_rttm_text(path: Union[Path, str]) -> Dict[str, List[Tuple[str, float, float]]]:
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
    """Reader class for 'rttm.scp'.

    Examples:
        SPEAKER file1 1 0 1023 <NA> <NA> spk1 <NA>
        SPEAKER file1 2 4000 3023 <NA> <NA> spk2 <NA>
        SPEAKER file1 3 500 4023 <NA> <NA> spk1 <NA>
        END     file1 <NA> 4023 <NA> <NA> <NA> <NA>

        This is an extend version of standard RTTM format for espnet.
        The difference including:
        1. Use sample number instead of absolute time
        2. has a END label to represent the duration of a recording
        3. replace duration (5th field) with end time
        (For standard RTTM,
            see https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)
        ...

        >>> reader = RttmReader('rttm')
        >>> spk_label = reader["file1"]

    """

    def __init__(
        self, fname: str,
    ):
        assert check_argument_types()
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
        return self.data.keys()
