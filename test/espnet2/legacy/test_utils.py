#!/usr/bin/env python3
import numpy as np
import pytest

from espnet2.legacy.utils.io_utils import SoundHDF5File


@pytest.mark.parametrize("fmt", ["flac", "wav"])
def test_sound_hdf5_file(tmpdir, fmt):
    valid = {
        "a": np.random.randint(-100, 100, 25, dtype=np.int16),
        "b": np.random.randint(-1000, 1000, 100, dtype=np.int16),
    }

    # Note: Specify the file format by extension
    p = tmpdir.join("test.{}.h5".format(fmt)).strpath
    f = SoundHDF5File(p, "a")

    for k, v in valid.items():
        f[k] = (v, 8000)

    for k, v in valid.items():
        t, r = f[k]
        assert r == 8000
        np.testing.assert_array_equal(t, v)
