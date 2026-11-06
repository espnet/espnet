import numpy as np
import pytest

from espnet2.fileio.rttm import RttmReader
from espnet2.fileio.vad_intervals import VadIntervalsReader
from espnet2.train.dataset import ESPnetDataset


def test_label_from_intervals(tmp_path):
    (tmp_path / "vad").write_text("w1 10 0 3 6 10\nw2 5\n")
    r = VadIntervalsReader(tmp_path / "vad")
    assert r["w1"].dtype == np.float32 and r["w1"].shape == (10, 1)
    assert r["w1"][:, 0].tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert not r["w2"].any() and r["w2"].shape == (5, 1)
    assert sorted(r) == ["w1", "w2"]


def test_odd_bounds_are_refused(tmp_path):
    (tmp_path / "vad").write_text("w1 10 0 3 6\n")
    with pytest.raises(ValueError):
        VadIntervalsReader(tmp_path / "vad")["w1"]


def test_matches_rttm_reader_slice(tmp_path):
    """RttmReader marks start..end inclusive; intervals are end exclusive."""
    (tmp_path / "rttm").write_text(
        "SPEAKER rec 1 2 4 <NA> <NA> rec <NA>\n"
        "SPEAKER rec 1 8 9 <NA> <NA> rec <NA>\n"
        "END rec <NA> <NA> 12 <NA> <NA> <NA> <NA>\n"
    )
    (tmp_path / "vad").write_text("rec 12 2 5 8 10\n")
    full = RttmReader(str(tmp_path / "rttm"))["rec"]
    np.testing.assert_array_equal(VadIntervalsReader(tmp_path / "vad")["rec"], full)


def test_dataset_type(tmp_path):
    (tmp_path / "vad").write_text("w1 4 1 3\n")
    ds = ESPnetDataset([(str(tmp_path / "vad"), "spk_labels", "vad_intervals")])
    _, d = ds["w1"]
    assert d["spk_labels"][:, 0].tolist() == [0, 1, 1, 0]
