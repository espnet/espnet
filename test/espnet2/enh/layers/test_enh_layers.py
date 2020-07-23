import pytest

import numpy as np
import torch
import torch_complex.functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.conv_beamformer import inv
from espnet2.enh.layers.conv_beamformer import signal_framing


@pytest.mark.parametrize("ch", [2, 4, 6, 8])
def test_inv(ch):
    torch.manual_seed(100)
    X = ComplexTensor(torch.rand(2, 3, ch, ch), torch.rand(2, 3, ch, ch))
    X = X + X.conj().transpose(-1, -2)
    assert FC.allclose(ComplexTensor(np.linalg.inv(X.numpy())), inv(X), atol=1e-4)


def test_signal_framing():
    # tap length = 1
    taps, delay = 0, 1
    X = ComplexTensor(torch.rand(2, 10, 6, 20), torch.rand(2, 10, 6, 20))
    X2 = signal_framing(X, taps + 1, 1, delay, do_padding=False)
    assert FC.allclose(X, X2.squeeze(-1))

    # tap length > 1, no padding
    taps, delay = 5, 3
    X = ComplexTensor(torch.rand(2, 10, 6, 20), torch.rand(2, 10, 6, 20))
    X2 = signal_framing(X, taps + 1, 1, delay, do_padding=False)
    assert X2.shape == torch.Size([2, 10, 6, 20 - taps - delay + 1, taps + 1])
    assert FC.allclose(X2[..., 0], X[..., : 20 - taps - delay + 1])

    # tap length > 1, padding
    taps, delay = 5, 3
    X = ComplexTensor(torch.rand(2, 10, 6, 20), torch.rand(2, 10, 6, 20))
    X2 = signal_framing(X, taps + 1, 1, delay, do_padding=True)
    assert X2.shape == torch.Size([2, 10, 6, 20, taps + 1])
    assert FC.allclose(X2[..., -1], X)
