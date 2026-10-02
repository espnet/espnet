from pathlib import Path

import numpy as np
import pytest

from espnet2.fileio.rttm import RttmReader, load_rttm_text


@pytest.fixture()
def rttm_file(tmp_path: Path):
    # The format egs2/TEMPLATE/asr1/pyscripts/utils/convert_rttm.py writes
    p = tmp_path / "espnet_rttm"
    with p.open("w") as f:
        f.write("SPEAKER abc 1 0 2 <NA> <NA> spk1 <NA>\n")
        f.write("SPEAKER abc 1 2 4 <NA> <NA> spk2 <NA>\n")
        f.write("SPEAKER abc 1 5 5 <NA> <NA> spk1 <NA>\n")
        f.write("SPEAKER def 1 1 2 <NA> <NA> spk3 <NA>\n")
        f.write("END abc <NA> <NA> 6 <NA> <NA> <NA> <NA>\n")
        f.write("END def <NA> <NA> 3 <NA> <NA> <NA> <NA>\n")
    return p


def test_load_rttm_text(rttm_file):
    assert load_rttm_text(rttm_file) == {
        "abc": (
            ["spk1", "spk2"],
            [("spk1", 0, 2), ("spk2", 2, 4), ("spk1", 5, 5)],
            6,
        ),
        "def": (["spk3"], [("spk3", 1, 2)], 3),
    }


def test_RttmReader(rttm_file):
    target = RttmReader(str(rttm_file))

    desired = {
        "abc": np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [1, 0]], dtype=float),
        "def": np.array([[0], [1], [1]], dtype=float),
    }
    for k in desired:
        np.testing.assert_array_equal(target[k], desired[k])
    assert len(target) == len(desired)
    assert tuple(target.keys()) == tuple(desired)
