import pytest
import torch
from packaging.version import parse as V

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.layers.dnn_beamformer import BEAMFORMER_TYPES
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
random_speech = torch.tensor(
    [
        [
            [0.026, 0.031, 0.023, 0.029, 0.026, 0.029, 0.028, 0.027],
            [0.027, 0.031, 0.023, 0.027, 0.026, 0.028, 0.027, 0.027],
            [0.026, 0.030, 0.023, 0.026, 0.025, 0.028, 0.028, 0.028],
            [0.024, 0.028, 0.024, 0.027, 0.024, 0.027, 0.030, 0.030],
            [0.025, 0.027, 0.025, 0.028, 0.023, 0.026, 0.031, 0.031],
            [0.027, 0.026, 0.025, 0.029, 0.022, 0.026, 0.032, 0.031],
            [0.028, 0.026, 0.024, 0.031, 0.023, 0.025, 0.031, 0.029],
            [0.029, 0.024, 0.023, 0.032, 0.023, 0.024, 0.030, 0.027],
            [0.028, 0.024, 0.023, 0.030, 0.023, 0.023, 0.028, 0.027],
            [0.029, 0.026, 0.023, 0.029, 0.025, 0.024, 0.027, 0.025],
            [0.029, 0.027, 0.024, 0.026, 0.025, 0.027, 0.025, 0.025],
            [0.029, 0.031, 0.026, 0.024, 0.028, 0.028, 0.024, 0.025],
            [0.030, 0.038, 0.029, 0.023, 0.035, 0.032, 0.024, 0.026],
            [0.029, 0.040, 0.030, 0.023, 0.039, 0.039, 0.025, 0.027],
            [0.028, 0.040, 0.032, 0.025, 0.041, 0.039, 0.026, 0.028],
            [0.028, 0.041, 0.039, 0.027, 0.044, 0.041, 0.029, 0.035],
        ],
        [
            [0.015, 0.021, 0.012, 0.006, 0.028, 0.021, 0.024, 0.018],
            [0.005, 0.034, 0.036, 0.017, 0.016, 0.037, 0.011, 0.029],
            [0.011, 0.029, 0.060, 0.029, 0.045, 0.035, 0.034, 0.018],
            [0.031, 0.036, 0.040, 0.037, 0.059, 0.032, 0.035, 0.029],
            [0.031, 0.031, 0.036, 0.029, 0.058, 0.035, 0.039, 0.045],
            [0.050, 0.038, 0.052, 0.052, 0.059, 0.044, 0.055, 0.045],
            [0.025, 0.054, 0.054, 0.047, 0.043, 0.059, 0.045, 0.060],
            [0.042, 0.056, 0.073, 0.029, 0.048, 0.063, 0.051, 0.049],
            [0.053, 0.048, 0.045, 0.052, 0.039, 0.045, 0.031, 0.053],
            [0.054, 0.044, 0.053, 0.031, 0.062, 0.050, 0.048, 0.046],
            [0.053, 0.036, 0.075, 0.046, 0.073, 0.052, 0.045, 0.030],
            [0.039, 0.025, 0.061, 0.046, 0.064, 0.032, 0.027, 0.033],
            [0.053, 0.032, 0.052, 0.033, 0.052, 0.029, 0.026, 0.017],
            [0.054, 0.034, 0.054, 0.033, 0.045, 0.043, 0.024, 0.018],
            [0.031, 0.025, 0.043, 0.016, 0.051, 0.040, 0.023, 0.030],
            [0.008, 0.023, 0.024, 0.019, 0.032, 0.024, 0.012, 0.027],
        ],
    ],
    dtype=torch.double,
)


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
@pytest.mark.parametrize("beamformer_type", BEAMFORMER_TYPES)
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
    elif num_spk == 1:
        if multi_source_wpe:
            # When num_spk == 1, `multi_source_wpe` has no effect
            return
        elif beamformer_type in (
            "lcmv",
            "lcmp",
            "wlcmp",
            "mvdr_tfs",
            "mvdr_tfs_souden",
        ):
            # only support multiple-source cases
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
        stft.output_dim,
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
        stft.output_dim,
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
    if not use_dnn_mask_for_wpe or multi_source_wpe:
        assert len(specs) == 1
    else:
        assert len(specs) == num_spk
    assert specs[0].shape == input_spectrum.shape
    if is_torch_1_9_plus and torch.is_complex(specs[0]):
        assert specs[0].dtype == torch.complex64
    else:
        assert specs[0].dtype == torch.float
    assert isinstance(others, dict)
    if use_dnn_mask_for_wpe:
        assert "mask_dereverb1" in others, others.keys()
        assert others["mask_dereverb1"].shape == specs[0].shape


@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("beamformer_type", BEAMFORMER_TYPES)
@pytest.mark.parametrize(
    "diagonal_loading, mask_flooring, use_torch_solver",
    [(True, True, True), (False, False, False)],
)
def test_neural_beamformer_bf_output(
    num_spk,
    use_noise_mask,
    beamformer_type,
    diagonal_loading,
    mask_flooring,
    use_torch_solver,
):
    if num_spk == 1 and beamformer_type in (
        "lcmv",
        "lcmp",
        "wlcmp",
        "mvdr_tfs",
        "mvdr_tfs_souden",
    ):
        # only support multiple-source cases
        return

    ch = 2
    inputs = random_speech[..., :ch].float()
    ilens = torch.LongTensor([16, 12])

    torch.random.manual_seed(0)
    stft = STFTEncoder(n_fft=8, hop_length=2)
    model = NeuralBeamformer(
        stft.output_dim,
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
        diagonal_loading=diagonal_loading,
        mask_flooring=mask_flooring,
        use_torch_solver=use_torch_solver,
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
        if is_torch_1_9_plus and torch.is_complex(specs[n - 1]):
            assert specs[n - 1].dtype == torch.complex64
        else:
            assert specs[n - 1].dtype == torch.float


def test_beamformer_net_invalid_bf_type():
    with pytest.raises(ValueError):
        NeuralBeamformer(10, use_beamformer=True, beamformer_type="fff")


def test_beamformer_net_invalid_loss_type():
    with pytest.raises(ValueError):
        NeuralBeamformer(10, loss_type="fff")
