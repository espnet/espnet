from pathlib import Path

import h5py
import kaldiio
import numpy as np
import pytest

from espnet2.fileio.kaldi_ark_scp import KaldiScpReader


@pytest.fixture()
def ark_scp(tmp_path: Path):
    ark_path1 = tmp_path / "a1.ark"
    scp_path1 = tmp_path / "a1.scp"
    array1 = np.random.randn(10, 10)
    array2 = np.random.randn(12, 10)
    with kaldiio.WriteHelper(f"ark,scp:{ark_path1},{scp_path1}") as f:
        f["abc"] = array1
        f["def"] = array2
    return ark_path1, scp_path1, (array1, array2)


def test_KaldiScpReader(ark_scp):
    ark_path1, scp_path1, (array1, array2) = ark_scp

    desired = {"abc": array1, "def": array2}
    target = KaldiScpReader(scp_path1)

    for k in desired:
        t = target[k]
        d = desired[k]
        np.testing.assert_array_equal(t, d)

    assert len(target) == len(desired)
    assert "abc" in target
    assert "def" in target
    assert tuple(target.keys()) == tuple(desired)
    assert tuple(target) == tuple(desired)
    assert target.get_path("abc") == str(ark_path1) + ":4"
    assert target.get_path("def") == str(ark_path1) + ":823"


def test_hdf5_KaldiScpReader(tmp_path: Path, ark_scp):
    ark_path1, scp_path1, (array1, array2) = ark_scp
    desired = {"abc": array1, "def": array2}

    f = h5py.File(tmp_path / "dummy.h5")
    target = KaldiScpReader(scp_path1)
    f["abc"] = target.get_path("abc")
    f["def"] = target.get_path("def")
    target = KaldiScpReader(f)

    for k in desired:
        t = target[k]
        d = desired[k]
        np.testing.assert_array_equal(t, d)

    assert len(target) == len(desired)
    assert "abc" in target
    assert "def" in target
    assert tuple(target.keys()) == tuple(desired)
    assert tuple(target) == tuple(desired)
