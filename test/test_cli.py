import numpy as np
import pytest

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import assert_scipy_wav_style
from espnet.utils.cli_writers import file_writer_helper


@pytest.mark.parametrize("filetype", ["mat", "hdf5", "sound.hdf5", "sound"])
def test_KaldiReader(tmpdir, filetype):
    ark = str(tmpdir.join("a.foo"))
    scp = str(tmpdir.join("a.scp"))
    fs = 16000

    with file_writer_helper(
        wspecifier=f"ark,scp:{ark},{scp}",
        filetype=filetype,
        write_num_frames="ark,t:out.txt",
        compress=False,
        compression_method=2,
        pcm_format="wav",
    ) as writer:
        if "sound" in filetype:
            aaa = np.random.randint(-10, 10, 100, dtype=np.int16)
            bbb = np.random.randint(-10, 10, 50, dtype=np.int16)
        else:
            aaa = np.random.randn(10, 10)
            bbb = np.random.randn(13, 5)
        if "sound" in filetype:
            writer["aaa"] = fs, aaa
            writer["bbb"] = fs, bbb
        else:
            writer["aaa"] = aaa
            writer["bbb"] = bbb
        valid = {"aaa": aaa, "bbb": bbb}

    # 1. Test ark read
    if filetype != "sound":
        for key, value in file_reader_helper(
            f"ark:{ark}", filetype=filetype, return_shape=False
        ):
            if "sound" in filetype:
                assert_scipy_wav_style(value)
                value = value[1]
            np.testing.assert_array_equal(value, valid[key])
    # 2. Test scp read
    for key, value in file_reader_helper(
        f"scp:{scp}", filetype=filetype, return_shape=False
    ):
        if "sound" in filetype:
            assert_scipy_wav_style(value)
            value = value[1]
        np.testing.assert_array_equal(value, valid[key])

    # 3. Test ark shape read
    if filetype != "sound":
        for key, value in file_reader_helper(
            f"ark:{ark}", filetype=filetype, return_shape=True
        ):
            if "sound" in filetype:
                value = value[1]
            np.testing.assert_array_equal(value, valid[key].shape)
    # 4. Test scp shape read
    for key, value in file_reader_helper(
        f"scp:{scp}", filetype=filetype, return_shape=True
    ):
        if "sound" in filetype:
            value = value[1]
        np.testing.assert_array_equal(value, valid[key].shape)
