import pytest

import numpy as np
import torch

from espnet2.enh.nets.tasnet import TasNet


@pytest.mark.parametrize("N", [5])
@pytest.mark.parametrize("L", [20])
@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("H", [10])
@pytest.mark.parametrize("P", [3])
@pytest.mark.parametrize("X", [8])
@pytest.mark.parametrize("R", [4])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("norm_type", ["BN", "gLN", "cLN"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mask_nonlinear", ["softmax", "relu"])
def test_tasnet_forward_backward(
    N, L, B, H, P, X, R, num_spk, norm_type, causal, mask_nonlinear,
):
    model = TasNet(
        N=N,
        L=L,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
        num_spk=num_spk,
        norm_type=norm_type,
        causal=causal,
        mask_nonlinear=mask_nonlinear,
    )

    est_speech, *_ = model(
        torch.randn(2, 100, requires_grad=True), ilens=torch.LongTensor([100, 80])
    )
    loss = sum([est.mean() for est in est_speech])
    loss.backward()


@pytest.mark.parametrize("N", [5, 10])
@pytest.mark.parametrize("L", [20])
@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("H", [10])
@pytest.mark.parametrize("P", [3])
@pytest.mark.parametrize("X", [8])
@pytest.mark.parametrize("R", [2, 4])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("norm_type", ["BN", "gLN", "cLN"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mask_nonlinear", ["softmax", "relu"])
def test_tasnet_cosistency(
    N, L, B, H, P, X, R, num_spk, norm_type, causal, mask_nonlinear,
):
    model = TasNet(
        N=N,
        L=L,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
        num_spk=num_spk,
        norm_type=norm_type,
        causal=causal,
        mask_nonlinear=mask_nonlinear,
    )
    model.eval()

    random_input_numpy = np.random.randn(2, 100)  # np.float64
    random_input_torch = (
        torch.from_numpy(random_input_numpy - 1.0).float() + 1.0
    )  # torch.float32
    random_input_numpy = torch.from_numpy(
        random_input_numpy.astype("float32")
    )  # np.float64-->np.float32-->torch.float32
    est_speech_numpy, *_ = model(random_input_numpy, ilens=torch.LongTensor([100, 80]))
    est_speech_torch, *_ = model(random_input_torch, ilens=torch.LongTensor([100, 80]))
    assert (est_speech_torch[0] - est_speech_numpy[0]).abs().mean() < 1e-5
    assert (
        np.abs((est_speech_torch[-1] - est_speech_numpy[-1]).detach().numpy()).mean()
        < 1e-5
    )


@pytest.mark.parametrize("N", [5])
@pytest.mark.parametrize("L", [20])
@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("H", [10])
@pytest.mark.parametrize("P", [3])
@pytest.mark.parametrize("X", [8])
@pytest.mark.parametrize("R", [4])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("norm_type", ["BN", "gLN", "cLN"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mask_nonlinear", ["softmax", "relu"])
def test_tasnet_output(
    N, L, B, H, P, X, R, num_spk, norm_type, causal, mask_nonlinear,
):
    inputs = torch.randn(2, 160)
    ilens = torch.LongTensor([160, 120])
    for num_spk in range(1, 3):
        model = TasNet(
            N=N,
            L=L,
            B=B,
            H=H,
            P=P,
            X=X,
            R=R,
            num_spk=num_spk,
            norm_type=norm_type,
            causal=causal,
            mask_nonlinear=mask_nonlinear,
        )
        specs, _, masks = model(inputs, ilens)
        assert isinstance(specs, list)
        assert isinstance(masks, dict)
        for n in range(num_spk):
            assert "spk{}".format(n + 1) in masks
            assert specs[n].shape == masks["spk{}".format(n + 1)].shape
            assert specs[n].shape == (2, 160)


def test_tasnet_invalid_norm_type():
    with pytest.raises(ValueError):
        TasNet(5, 20, 5, 10, 3, 8, 4, 2, norm_type="fff")


def test_tasnet_invalid_mask_nonlinear():
    with pytest.raises(ValueError):
        TasNet(5, 20, 5, 10, 3, 8, 4, 2, mask_nonlinear="fff")
