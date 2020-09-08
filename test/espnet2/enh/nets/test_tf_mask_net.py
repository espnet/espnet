import pytest

import numpy as np
import torch

from espnet2.enh.nets.tf_mask_net import TFMaskingNet


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length",
    [(8, None, 2)],
)
@pytest.mark.parametrize("rnn_type", ["blstm"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("utt_mvn", [True, False])
@pytest.mark.parametrize("mask_type", ["IRM"])
@pytest.mark.parametrize("loss_type", ["mask_mse", "magnitude", "spectrum"])
def test_tf_mask_net_forward_backward(
    n_fft,
    win_length,
    hop_length,
    rnn_type,
    layer,
    unit,
    dropout,
    num_spk,
    nonlinear,
    utt_mvn,
    mask_type,
    loss_type,
):
    model = TFMaskingNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        rnn_type=rnn_type,
        layer=layer,
        unit=unit,
        dropout=dropout,
        num_spk=num_spk,
        nonlinear=nonlinear,
        utt_mvn=utt_mvn,
        mask_type=mask_type,
        loss_type=loss_type,
    )
    model.train()

    if loss_type.startswith("mask"):
        # mask backward
        est_speech, flens, masks = model(
            torch.randn(2, 16, requires_grad=True), ilens=torch.LongTensor([16, 12])
        )
        loss = sum([masks[key].mean() for key in masks])
        loss.backward()
    else:
        # spectrums backward
        est_speech, flens, masks = model(
            torch.randn(2, 16, requires_grad=True), ilens=torch.LongTensor([16, 12])
        )
        loss = sum([abs(est).mean() for est in est_speech])
        loss.backward()


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length",
    [(8, None, 2)],
)
@pytest.mark.parametrize("rnn_type", ["blstm"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("utt_mvn", [True, False])
@pytest.mark.parametrize("mask_type", ["IRM"])
@pytest.mark.parametrize("loss_type", ["mask_mse", "magnitude", "spectrum"])
def test_tf_mask_net_consistency(
    n_fft,
    win_length,
    hop_length,
    rnn_type,
    layer,
    unit,
    dropout,
    num_spk,
    nonlinear,
    utt_mvn,
    mask_type,
    loss_type,
):
    model = TFMaskingNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        rnn_type=rnn_type,
        layer=layer,
        unit=unit,
        dropout=dropout,
        num_spk=num_spk,
        nonlinear=nonlinear,
        utt_mvn=utt_mvn,
        mask_type=mask_type,
        loss_type=loss_type,
    )

    model.eval()

    random_input_numpy = np.random.randn(2, 16)  # np.float64
    random_input_torch = (
        torch.from_numpy(random_input_numpy - 1.0).float() + 1.0
    )  # torch.float32
    random_input_numpy = torch.from_numpy(
        random_input_numpy.astype("float32")
    )  # np.float64-->np.float32-->torch.float32
    est_speech_numpy, flens, masks = model(
        random_input_numpy, ilens=torch.LongTensor([16, 12])
    )

    est_speech_torch, flens, masks = model(
        random_input_torch, ilens=torch.LongTensor([16, 12])
    )
    assert (est_speech_torch[0] - est_speech_numpy[0]).abs().mean() < 1e-5
    assert (
        np.abs(
            (est_speech_torch[-1] - est_speech_numpy[-1]).detach().real.numpy()
        ).mean()
        < 1e-5
    )


def test_tf_mask_net_output():
    inputs = torch.randn(2, 16)
    ilens = torch.LongTensor([16, 12])
    for num_spk in range(1, 3):
        model = TFMaskingNet(
            n_fft=8,
            win_length=None,
            hop_length=2,
            rnn_type="blstm",
            layer=3,
            unit=8,
            dropout=0.0,
            num_spk=num_spk,
        )
        model.eval()
        specs, _, masks = model(inputs, ilens)
        assert isinstance(specs, list)
        assert isinstance(masks, dict)
        for n in range(num_spk):
            assert "spk{}".format(n + 1) in masks
            assert specs[n].shape == masks["spk{}".format(n + 1)].shape


def test_tf_mask_net_invalid_norm_type():
    with pytest.raises(ValueError):
        TFMaskingNet(
            n_fft=8,
            win_length=None,
            hop_length=2,
            rnn_type="blstm",
            layer=3,
            unit=8,
            dropout=0.0,
            num_spk=2,
            nonlinear="fff",
        )


def test_tf_mask_net_invalid_loss_type():
    with pytest.raises(ValueError):
        TFMaskingNet(loss_type="fff")
