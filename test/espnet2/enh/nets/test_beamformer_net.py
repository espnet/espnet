import pytest

import numpy as np
import torch
from torch_complex import functional as FC

from espnet2.enh.nets.beamformer_net import BeamformerNet


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length",
    [(8, None, 2)],
)
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("normalize_input", [True])
@pytest.mark.parametrize("mask_type", ["IPM^2"])
@pytest.mark.parametrize("loss_type", ["mask_mse", "spectrum"])
@pytest.mark.parametrize("use_wpe", [True])
@pytest.mark.parametrize("wnet_type", ["lstm"])
@pytest.mark.parametrize("wlayers", [2])
@pytest.mark.parametrize("wunits", [2])
@pytest.mark.parametrize("wprojs", [2])
@pytest.mark.parametrize("taps", [2])
@pytest.mark.parametrize("delay", [3])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [False])
@pytest.mark.parametrize("multi_source_wpe", [True, False])
@pytest.mark.parametrize("use_beamformer", [True])
@pytest.mark.parametrize("bnet_type", ["lstm"])
@pytest.mark.parametrize("blayers", [2])
@pytest.mark.parametrize("bunits", [2])
@pytest.mark.parametrize("bprojs", [2])
@pytest.mark.parametrize("badim", [2])
@pytest.mark.parametrize("ref_channel", [-1, 0])
@pytest.mark.parametrize("use_noise_mask", [True])
@pytest.mark.parametrize("bnonlinear", ["sigmoid", "relu", "tanh", "crelu"])
@pytest.mark.parametrize(
    "beamformer_type",
    [
        "mvdr_souden",
        "mpdr_souden",
        "wmpdr_souden",
        "wpd_souden",
        "mvdr",
        "mpdr",
        "wmpdr",
        "wpd",
    ],
)
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
    taps,
    delay,
    use_dnn_mask_for_wpe,
    multi_source_wpe,
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
    # Skip some cases
    if num_spk > 1 and use_wpe and use_beamformer:
        if not multi_source_wpe:
            # Single-source WPE is not supported with beamformer in multi-speaker cases
            return
    elif num_spk == 1 and multi_source_wpe:
        # When num_spk == 1, `multi_source_wpe` has no effect
        return
    if bnonlinear != "sigmoid" and (
        beamformer_type != "mvdr_souden" or multi_source_wpe
    ):
        # only test different nonlinear layers with MVDR_Souden
        return

    model = BeamformerNet(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        num_spk=num_spk,
        normalize_input=normalize_input,
        train_mask_only=True,
        mask_type=mask_type,
        loss_type=loss_type,
        use_wpe=use_wpe,
        wnet_type=wnet_type,
        wlayers=wlayers,
        wunits=wunits,
        wprojs=wprojs,
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
        rtf_iterations=2,
        shared_power=True,
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
@pytest.mark.parametrize("use_wpe", [False])
@pytest.mark.parametrize("wnet_type", ["lstm"])
@pytest.mark.parametrize("wlayers", [2])
@pytest.mark.parametrize("wunits", [2])
@pytest.mark.parametrize("wprojs", [2])
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
@pytest.mark.parametrize("use_noise_mask", [True])
@pytest.mark.parametrize("beamformer_type", ["mvdr_souden"])
def test_beamformer_net_consistency(
    n_fft,
    win_length,
    hop_length,
    num_spk,
    normalize_input,
    mask_type,
    use_wpe,
    wnet_type,
    wlayers,
    wunits,
    wprojs,
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
        train_mask_only=True,
        mask_type=mask_type,
        use_wpe=use_wpe,
        wnet_type=wnet_type,
        wlayers=wlayers,
        wunits=wunits,
        wprojs=wprojs,
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
    assert FC.allclose(est_speech_torch[0], est_speech_numpy[0])
    assert FC.allclose(est_speech_torch[-1], est_speech_numpy[-1])
    for est in est_speech_torch:
        assert est.dtype == torch.float

    for ps in est_speech_torch:
        enh_waveform = model.stft.inverse(ps, torch.LongTensor([16, 12]))[0]

        assert enh_waveform.shape == random_input_torch.shape[:-1], (
            enh_waveform.shape,
            random_input_torch.shape,
        )


@pytest.mark.parametrize("ch", [1, 2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("multi_source_wpe", [True, False])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [True, False])
def test_beamformer_net_wpe_output(ch, num_spk, multi_source_wpe, use_dnn_mask_for_wpe):
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
        multi_source_wpe=multi_source_wpe,
        wlayers=2,
        wunits=2,
        wprojs=2,
        taps=5,
        delay=3,
        use_beamformer=False,
    )
    model.eval()
    specs, _, masks = model(inputs, ilens)
    assert isinstance(specs, list)
    assert len(specs) == num_spk if multi_source_wpe else 1
    assert specs[0].shape[0] == 2  # batch size
    assert specs[0].dtype == torch.float
    assert isinstance(masks, dict)
    if use_dnn_mask_for_wpe:
        assert "dereverb" in masks
        assert masks["dereverb"].shape == specs[0].shape


@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("use_noise_mask", [True, False])
def test_beamformer_net_bf_output(num_spk, use_noise_mask):
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
        blayers=2,
        bunits=2,
        bprojs=2,
        badim=2,
        use_noise_mask=use_noise_mask,
    )
    model.eval()
    specs, _, masks = model(inputs, ilens)
    assert isinstance(masks, dict)
    assert "noise1" in masks
    assert masks["noise1"].shape == masks["spk1"].shape
    assert isinstance(specs, list)
    assert len(specs) == num_spk
    for n in range(1, num_spk + 1):
        assert "spk{}".format(n) in masks
        assert masks["spk{}".format(n)].shape[-2] == ch
        assert specs[n - 1].shape == masks["spk{}".format(n)][..., 0, :].shape
        assert specs[n - 1].dtype == torch.float


def test_beamformer_net_invalid_bf_type():
    with pytest.raises(ValueError):
        BeamformerNet(use_beamformer=True, beamformer_type="fff")


def test_beamformer_net_invalid_loss_type():
    with pytest.raises(ValueError):
        BeamformerNet(loss_type="fff")
