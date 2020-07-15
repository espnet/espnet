import pytest
import torch
import numpy as np

from espnet2.enh.nets.beamformer_net import BeamformerNet


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length", [(8, None, 2)],
)
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("use_wpe, taps", [(True, 3), (False, 0)])
@pytest.mark.parametrize("wnet_type", ["blstmp"])
@pytest.mark.parametrize("wlayers", [3])
@pytest.mark.parametrize("wunits", [8])
@pytest.mark.parametrize("wprojs", [10])
@pytest.mark.parametrize("wdropout_rate", [0.0, 0.2])
@pytest.mark.parametrize("delay", [3])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [True, False])
@pytest.mark.parametrize("use_beamformer", [True])
@pytest.mark.parametrize("bnet_type", ["blstmp"])
@pytest.mark.parametrize("blayers", [3])
@pytest.mark.parametrize("bunits", [8])
@pytest.mark.parametrize("bprojs", [10])
@pytest.mark.parametrize("badim", [10])
@pytest.mark.parametrize("ref_channel", [-1, 0])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("beamformer_type", ["mvdr", "mpdr", "wpd"])
@pytest.mark.parametrize("bdropout_rate", [0.0, 0.2])
def test_beamformer_net_forward_backward(
    n_fft,
    win_length,
    hop_length,
    num_spk,
    normalize_input,
    use_wpe,
    wnet_type,
    wlayers,
    wunits,
    wprojs,
    wdropout_rate,
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
    bdropout_rate,
):
    model = BeamformerNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        num_spk=num_spk,
        normalize_input=normalize_input,
        use_wpe=use_wpe,
        wnet_type=wnet_type,
        wlayers=wlayers,
        wunits=wunits,
        wprojs=wprojs,
        wdropout_rate=wdropout_rate,
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
        bdropout_rate=bdropout_rate,
    )

    est_speech, *_ = model(
        torch.randn(2, 16, 2, requires_grad=True), ilens=torch.LongTensor([16, 12])
    )
    loss = sum([est.mean() for est in est_speech])
    loss.backward()


# Cause many errors.
@pytest.mark.parametrize(
    "n_fft, win_length, hop_length", [(8, None, 2)],
)
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("use_wpe, taps", [(True, 3), (False, 0)])
@pytest.mark.parametrize("wnet_type", ["blstmp"])
@pytest.mark.parametrize("wlayers", [3])
@pytest.mark.parametrize("wunits", [8])
@pytest.mark.parametrize("wprojs", [10])
@pytest.mark.parametrize("wdropout_rate", [0.0, 0.2])
@pytest.mark.parametrize("delay", [3])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [True, False])
@pytest.mark.parametrize("use_beamformer", [True])
@pytest.mark.parametrize("bnet_type", ["blstmp"])
@pytest.mark.parametrize("blayers", [3])
@pytest.mark.parametrize("bunits", [8])
@pytest.mark.parametrize("bprojs", [10])
@pytest.mark.parametrize("badim", [10])
@pytest.mark.parametrize("ref_channel", [-1, 0])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("beamformer_type", ["mvdr", "mpdr", "wpd"])
@pytest.mark.parametrize("bdropout_rate", [0.0, 0.2])
def test_beamformer_net_consistency(
    n_fft,
    win_length,
    hop_length,
    num_spk,
    normalize_input,
    use_wpe,
    wnet_type,
    wlayers,
    wunits,
    wprojs,
    wdropout_rate,
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
    bdropout_rate,
):
    model = BeamformerNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        num_spk=num_spk,
        normalize_input=normalize_input,
        use_wpe=use_wpe,
        wnet_type=wnet_type,
        wlayers=wlayers,
        wunits=wunits,
        wprojs=wprojs,
        wdropout_rate=wdropout_rate,
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
        bdropout_rate=bdropout_rate,
    )
    model.eval()

    random_input_numpy = np.random.randn(2, 16, 2)  # np.float64
    random_input_torch = torch.from_numpy(random_input_numpy).float()
    random_input_numpy = torch.from_numpy(
        random_input_numpy.astype("float32")
    )  # np.float64-->np.float32-->torch.float32
    est_speech_numpy, *_ = model(random_input_numpy, ilens=torch.LongTensor([16, 12]))

    est_speech_torch, *_ = model(random_input_torch, ilens=torch.LongTensor([16, 12]))
    assert (est_speech_torch[0] - est_speech_numpy[0]).abs().mean() < 1e-3
    assert (
        np.abs((est_speech_torch[-1] - est_speech_numpy[-1]).detach().numpy()).mean()
        < 1e-3
    )


def test_beamformer_net_output():
    inputs = torch.randn(2, 16, 2)
    ilens = torch.LongTensor([16, 12])
    for num_spk in range(1, 3):
        model = BeamformerNet(
            n_fft=8, hop_length=2, num_spk=num_spk, use_wpe=False, use_beamformer=True
        )
        specs, _, masks = model(inputs, ilens)
        assert isinstance(specs, list)
        assert len(specs) == num_spk
        assert isinstance(masks, dict)
        for n in range(1, num_spk + 1):
            assert "spk{}".format(n) in masks
            assert specs[n].shape == masks["spk{}".format(n)].shape


def test_beamformer_net_invalid_bf_type():
    with pytest.raises(ValueError):
        BeamformerNet(use_beamformer=True, beamformer_type="fff")
