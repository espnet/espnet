import pytest
import torch

from espnet2.asr.frontend.nets.tf_mask_net import TFMaskingNet


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length", [(8, None, 2)],
)
@pytest.mark.parametrize("rnn_type", ["blstm",])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8,])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("none_linear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("utt_mvn", [True, False])
def test_tf_mask_net_forward_backward(
    n_fft,
    win_length,
    hop_length,
    rnn_type,
    layer,
    unit,
    dropout,
    num_spk,
    none_linear,
    utt_mvn,
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
        none_linear=none_linear,
        utt_mvn=utt_mvn,
    )

    # mask backward
    est_speech, flens, masks = model(
        torch.randn(2, 16, requires_grad=True), ilens=torch.LongTensor([16, 12])
    )
    loss = sum([masks[key].mean() for key in masks])
    loss.backward()

    # spectrums backward
    est_speech, flens, masks = model(
        torch.randn(2, 16, requires_grad=True), ilens=torch.LongTensor([16, 12])
    )
    loss = sum([abs(est).mean() for est in est_speech])
    loss.backward()


def test_tf_mask_net_output():
    inputs = torch.randn(2, 16)
    ilens = torch.LongTensor([16, 12])
    for num_spk in range(1, 3):
        model = TFMaskingNet(8, None, 2, "blstm", 3, 8, 0.0, num_spk)
        specs, _, masks = model(inputs, ilens)
        assert isinstance(specs, list)
        assert isinstance(masks, dict)
        for n in range(num_spk):
            assert "spk{}".format(n+1) in masks
            assert specs[n].shape == masks["spk{}".format(n+1)].shape


def test_tf_mask_net_invalid_norm_type():
    with pytest.raises(ValueError):
        TFMaskingNet(8, None, 2, "blstm", 3, 8, 0.0, 2, none_linear="fff")
