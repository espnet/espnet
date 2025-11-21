import kaldiio
import numpy as np
import pytest

from espnet2.legacy.transform.spectrogram import logmelspectrogram
from espnet2.legacy.transform.transformation import Transformation


@pytest.mark.execution_timeout(10)
def test_preprocessing(tmpdir):
    cmvn_ark = str(tmpdir.join("cmvn.ark"))
    kwargs = {
        "process": [
            {"type": "fbank", "n_mels": 80, "fs": 16000, "n_fft": 1024, "n_shift": 512},
        ],
        "mode": "sequential",
    }

    # Creates cmvn_ark
    samples = np.random.randn(100, 80)
    stats = np.empty((2, 81), dtype=np.float32)
    stats[0, :80] = samples.sum(axis=0)
    stats[1, :80] = (samples**2).sum(axis=0)
    stats[0, -1] = 100.0
    stats[1, -1] = 0.0
    kaldiio.save_mat(cmvn_ark, stats)

    bs = 1
    xs = [np.random.randn(1000).astype(np.float32) for _ in range(bs)]
    preprocessing = Transformation(kwargs)
    processed_xs = preprocessing(xs)

    for idx, x in enumerate(xs):
        opt = dict(kwargs["process"][0])
        opt.pop("type")
        x = logmelspectrogram(x, **opt)

        np.testing.assert_allclose(processed_xs[idx], x)
