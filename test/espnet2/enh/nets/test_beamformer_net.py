from distutils.version import LooseVersion

import pytest
import torch

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from test.espnet2.enh.layers.test_enh_layers import random_speech

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")


@pytest.mark.parametrize(
    "n_fft, win_length, hop_length",
    [(8, None, 2)],
)
@pytest.mark.parametrize("num_spk", [1, 2])
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
def test_neural_beamformer_forward_backward(
    n_fft,
    win_length,
    hop_length,
    num_spk,
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
    bnonlinear,
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

    # ensures reproducibility and reversibility in the matrix inverse computation
    torch.random.manual_seed(0)
    stft = STFTEncoder(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    model = NeuralBeamformer(
        num_spk=num_spk,
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
    inputs = random_speech[..., :2].float()
    ilens = torch.LongTensor([16, 12])
    input_spectrum, flens = stft(inputs, ilens)
    est_speech, flens, others = model(input_spectrum, flens)
    if loss_type.startswith("mask"):
        assert est_speech is None
        loss = sum([abs(m).mean() for m in others.values()])
    else:
        loss = sum([abs(est).mean() for est in est_speech])
    loss.backward()


@pytest.mark.parametrize("ch", [1, 2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("multi_source_wpe", [True, False])
@pytest.mark.parametrize("use_dnn_mask_for_wpe", [True, False])
def test_neural_beamformer_wpe_output(
    ch, num_spk, multi_source_wpe, use_dnn_mask_for_wpe
):
    torch.random.manual_seed(0)
    inputs = torch.randn(2, 16, ch) if ch > 1 else torch.randn(2, 16)
    inputs = inputs.float()
    ilens = torch.LongTensor([16, 12])
    stft = STFTEncoder(n_fft=8, hop_length=2)
    model = NeuralBeamformer(
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
    input_spectrum, flens = stft(inputs, ilens)
    specs, _, others = model(input_spectrum, flens)
    assert isinstance(specs, list)
    assert len(specs) == (1 if multi_source_wpe else num_spk)
    if ch > 1:
        assert specs[0].shape == input_spectrum[..., 0, :].shape
    else:
        assert specs[0].shape == input_spectrum.shape
    assert specs[0].dtype == torch.float
    assert isinstance(others, dict)
    if use_dnn_mask_for_wpe:
        assert "mask_dereverb1" in others, others.keys()
        assert others["mask_dereverb1"].shape == specs[0].shape


@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("use_noise_mask", [True, False])
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
def test_neural_beamformer_bf_output(num_spk, use_noise_mask, beamformer_type):
    ch = 2
    inputs = random_speech[..., :ch].float()
    ilens = torch.LongTensor([16, 12])

    torch.random.manual_seed(0)
    stft = STFTEncoder(n_fft=8, hop_length=2)
    model = NeuralBeamformer(
        num_spk=num_spk,
        use_wpe=False,
        taps=2,
        delay=3,
        use_beamformer=True,
        blayers=2,
        bunits=2,
        bprojs=2,
        badim=2,
        use_noise_mask=use_noise_mask,
        beamformer_type=beamformer_type,
    )
    model.eval()
    input_spectrum, flens = stft(inputs, ilens)
    specs, _, others = model(input_spectrum, flens)
    assert isinstance(others, dict)
    if use_noise_mask:
        assert "mask_noise1" in others
        assert others["mask_noise1"].shape == others["mask_spk1"].shape
    assert isinstance(specs, list)
    assert len(specs) == num_spk
    for n in range(1, num_spk + 1):
        assert "mask_spk{}".format(n) in others, others.keys()
        assert others["mask_spk{}".format(n)].shape[-2] == ch
        assert specs[n - 1].shape == others["mask_spk{}".format(n)][..., 0, :].shape
        assert specs[n - 1].shape == input_spectrum[..., 0, :].shape
        assert specs[n - 1].dtype == torch.float


def test_beamformer_net_invalid_bf_type():
    with pytest.raises(ValueError):
        NeuralBeamformer(use_beamformer=True, beamformer_type="fff")


def test_beamformer_net_invalid_loss_type():
    with pytest.raises(ValueError):
        NeuralBeamformer(loss_type="fff")
