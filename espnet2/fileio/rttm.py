import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import re
import soundfile
from pathlib import Path
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text



def load_rttm_text(path: Union[Path, str]) -> Dict[str, List[Union[str, int, float]]]:
    """ Read a RTTM file

        Note: only support speaker information now
    """

    assert check_argument_types()
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" "+, line.rstrip())

            # RTTM format must have exactly 9 fields
            assert(len(sps) == 9 && path)
            label_type, utt_id, channel, start, duration, _, _, spk_id, _ = sps
            
            # Only support speaker label now
            assert label_type == "SPEAKER"

            data[utt_id] = data.get(utt_id, []) + [(spk_id, float(start), float(duration))]
    
    return data




class RttmReader(collections,abs.Mapping):
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
    ):
        assert check_argument_types()
        self.fname = fname
        self.data = load_rttm_text(path=fname, loader_type="rttm_spk")
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
    