import pytest

import numpy as np
import torch

from espnet2.enh.nets.beamformer_net import BeamformerNet


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length",
    [(8, None, 2)],
)
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("normalize_input", [True])
@pytest.mark.parametrize("mask_type", ["IPM^2"])
@pytest.mark.parametrize("loss_type", ["mask_mse", "spectrum"])
@pytest.mark.parametrize("use_wpe", [False])
@pytest.mark.parametrize("wnet_type", ["lstm"])
@pytest.mark.parametrize("wlayers", [2])
@pytest.mark.parametrize("wunits", [2])
@pytest.mark.parametrize("wprojs", [2])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
@pytest.mark.parametrize("taps", [2])
@pytest.mark.parametrize("delay", [3])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [False])
@pytest.mark.parametrize("use_beamformer", [True])
@pytest.mark.parametrize("bnet_type", ["lstm"])
@pytest.mark.parametrize("blayers", [2])
@pytest.mark.parametrize("bunits", [2])
@pytest.mark.parametrize("bprojs", [2])
@pytest.mark.parametrize("badim", [2])
@pytest.mark.parametrize("ref_channel", [-1, 0])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("beamformer_type", ["mvdr", "mpdr", "wpd"])
def test_beamformer_net_forward_backward(
    n_fft,
    win_length,
    hop_length,
    num_spk,
    normalize_input,
    mask_type,
    loss_type,
    use_wpe,
    wnet_type,
    wlayers,
    wunits,
    wprojs,
    dropout_rate,
    taps,
    delay,
    use_dnn_mask_for_wpe,
    use_beamformer,
    bnet_type,
    blayers,
    bunits,
    bprojs,
    badim,
    ref_channel,
    use_noise_mask,
    beamformer_type,
):
    model = BeamformerNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        num_spk=num_spk,
        normalize_input=normalize_input,
        mask_type=mask_type,
        loss_type=loss_type,
        use_wpe=use_wpe,
        wnet_type=wnet_type,
        wlayers=wlayers,
        wunits=wunits,
        wprojs=wprojs,
        wdropout_rate=dropout_rate,
        taps=taps,
        delay=delay,
        use_dnn_mask_for_wpe=use_dnn_mask_for_wpe,
        use_beamformer=use_beamformer,
        bnet_type=bnet_type,
        blayers=blayers,
        bunits=bunits,
        bprojs=bprojs,
        badim=badim,
        ref_channel=ref_channel,
        use_noise_mask=use_noise_mask,
        beamformer_type=beamformer_type,
        bdropout_rate=dropout_rate,
    )

    model.train()
    est_speech, flens, masks = model(
        torch.randn(2, 16, 2, requires_grad=True), ilens=torch.LongTensor([16, 12])
    )
    if loss_type.startswith("mask"):
        assert est_speech is None
        loss = sum([abs(m).mean() for m in masks.values()])
    else:
        loss = sum([abs(est).mean() for est in est_speech])
    loss.backward()


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length",
    [(8, None, 2)],
)
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("normalize_input", [True])
@pytest.mark.parametrize("mask_type", ["IPM^2"])
@pytest.mark.parametrize("loss_type", ["mask_mse", "spectrum"])
@pytest.mark.parametrize("use_wpe", [False])
@pytest.mark.parametrize("wnet_type", ["lstm"])
@pytest.mark.parametrize("wlayers", [2])
@pytest.mark.parametrize("wunits", [2])
@pytest.mark.parametrize("wprojs", [2])
@pytest.mark.parametrize("dropout_rate", [0.0])
@pytest.mark.parametrize("taps", [2])
@pytest.mark.parametrize("delay", [3])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [False])
@pytest.mark.parametrize("use_beamformer", [True])
@pytest.mark.parametrize("bnet_type", ["lstm"])
@pytest.mark.parametrize("blayers", [2])
@pytest.mark.parametrize("bunits", [2])
@pytest.mark.parametrize("bprojs", [2])
@pytest.mark.parametrize("badim", [10])
@pytest.mark.parametrize("ref_channel", [-1, 0])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("beamformer_type", ["mvdr", "mpdr", "wpd"])
def test_beamformer_net_consistency(
    n_fft,
    win_length,
    hop_length,
    num_spk,
    normalize_input,
    mask_type,
    loss_type,
    use_wpe,
    wnet_type,
    wlayers,
    wunits,
    wprojs,
    dropout_rate,
    taps,
    delay,
    use_dnn_mask_for_wpe,
    use_beamformer,
    bnet_type,
    blayers,
    bunits,
    bprojs,
    badim,
    ref_channel,
    use_noise_mask,
    beamformer_type,
):
    model = BeamformerNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        num_spk=num_spk,
        normalize_input=normalize_input,
        mask_type=mask_type,
        loss_type=loss_type,
        use_wpe=use_wpe,
        wnet_type=wnet_type,
        wlayers=wlayers,
        wunits=wunits,
        wprojs=wprojs,
        wdropout_rate=dropout_rate,
        taps=taps,
        delay=delay,
        use_dnn_mask_for_wpe=use_dnn_mask_for_wpe,
        use_beamformer=use_beamformer,
        bnet_type=bnet_type,
        blayers=blayers,
        bunits=bunits,
        bprojs=bprojs,
        badim=badim,
        ref_channel=ref_channel,
        use_noise_mask=use_noise_mask,
        beamformer_type=beamformer_type,
        bdropout_rate=dropout_rate,
    )

    model.eval()

    random_input_numpy = np.random.randn(2, 16, 2)  # np.float64
    random_input_torch = torch.from_numpy(random_input_numpy).float()
    random_input_numpy = torch.from_numpy(
        random_input_numpy.astype("float32")
    )  # np.float64-->np.float32-->torch.float32

    # ensures reproducibility in the matrix inverse computation
    torch.random.manual_seed(0)
    est_speech_numpy, *_ = model(random_input_numpy, ilens=torch.LongTensor([16, 12]))

    torch.random.manual_seed(0)
    est_speech_torch, *_ = model(random_input_torch, ilens=torch.LongTensor([16, 12]))
    assert torch.allclose(est_speech_torch[0], est_speech_numpy[0])
    assert torch.allclose(est_speech_torch[-1], est_speech_numpy[-1])
    for est in est_speech_torch:
        assert est.dtype == torch.float


@pytest.mark.parametrize("ch", [1, 2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [True, False])
def test_beamformer_net_wpe_output(ch, num_spk, use_dnn_mask_for_wpe):
    torch.random.manual_seed(0)
    inputs = torch.randn(2, 16, ch) if ch > 1 else torch.randn(2, 16)
    inputs = inputs.float()
    ilens = torch.LongTensor([16, 12])
    model = BeamformerNet(
        n_fft=8,
        hop_length=2,
        num_spk=num_spk,
        use_wpe=True,
        use_dnn_mask_for_wpe=use_dnn_mask_for_wpe,
        taps=5,
        delay=3,
        use_beamformer=False,
    )
    model.eval()
    spec, _, masks = model(inputs, ilens)
    assert spec.shape[0] == 2  # batch size
    assert spec.shape[-1] == 2  # real and imag
    assert spec.dtype == torch.float
    assert isinstance(masks, dict)
    if use_dnn_mask_for_wpe:
        assert "dereverb" in masks
        assert masks["dereverb"].shape == spec.shape[:-1]


@pytest.mark.parametrize("num_spk", [1, 2])
def test_beamformer_net_bf_output(num_spk):
    ch = 2
    inputs = torch.randn(2, 16, ch)
    inputs = inputs.float()
    ilens = torch.LongTensor([16, 12])
    model = BeamformerNet(
        n_fft=8,
        hop_length=2,
        num_spk=num_spk,
        use_wpe=False,
        use_beamformer=True,
        use_noise_mask=True,
    )
    model.eval()
    specs, _, masks = model(inputs, ilens)
    assert isinstance(masks, dict)
    assert "noise1" in masks
    assert masks["noise1"].shape == masks["spk1"].shape
    if num_spk > 1:
        assert isinstance(specs, list)
        assert len(specs) == num_spk
        for n in range(1, num_spk + 1):
            assert "spk{}".format(n) in masks
            assert masks["spk{}".format(n)].shape[-2] == ch
            assert specs[n - 1].shape[:-1] == masks["spk{}".format(n)][..., 0, :].shape
            assert specs[n - 1].shape[-1] == 2  # real and imag
            assert specs[n - 1].dtype == torch.float
    else:
        assert isinstance(specs, torch.Tensor)
        assert "spk1" in masks
        assert masks["spk1"].shape[-2] == ch
        assert specs.shape[:-1] == masks["spk1"][..., 0, :].shape
        assert specs.shape[-1] == 2  # real and imag
        assert specs.dtype == torch.float


def test_beamformer_net_invalid_bf_type():
    with pytest.raises(ValueError):
        BeamformerNet(use_beamformer=True, beamformer_type="fff")


def test_beamformer_net_invalid_loss_type():
    with pytest.raises(ValueError):
        BeamformerNet(loss_type="fff")
