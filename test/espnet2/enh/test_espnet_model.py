from distutils.version import LooseVersion

import pytest
import torch

from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")


stft_encoder = STFTEncoder(
    n_fft=28,
    hop_length=16,
)

stft_decoder = STFTDecoder(
    n_fft=28,
    hop_length=16,
)

conv_encoder = ConvEncoder(
    channel=15,
    kernel_size=32,
    stride=16,
)

conv_decoder = ConvDecoder(
    channel=15,
    kernel_size=32,
    stride=16,
)

rnn_separator = RNNSeparator(
    input_dim=15,
    layer=1,
    unit=10,
)

dprnn_separator = DPRNNSeparator(input_dim=15, layer=1, unit=10, segment_size=4)

tcn_separator = TCNSeparator(
    input_dim=15,
    layer=2,
    stack=1,
    bottleneck_dim=10,
    hidden_dim=10,
    kernel=3,
)

transformer_separator = TransformerSeparator(
    input_dim=15,
    adim=8,
    aheads=2,
    layers=2,
    linear_units=10,
)


@pytest.mark.parametrize(
    "encoder, decoder", [(stft_encoder, stft_decoder), (conv_encoder, conv_decoder)]
)
@pytest.mark.parametrize(
    "separator", [rnn_separator, dprnn_separator, tcn_separator, transformer_separator]
)
@pytest.mark.parametrize(
    "loss_type",
    ["si_snr", "ci_sdr", "mask_mse", "magnitude", "spectrum", "spectrum_log"],
)
@pytest.mark.parametrize("stft_consistency", [True, False])
@pytest.mark.parametrize("mask_type", ["IBM", "IRM", "IAM", "PSM", "NPSM", "PSM^2"])
@pytest.mark.parametrize("training", [True, False])
def test_single_channel_model(
    encoder, decoder, separator, stft_consistency, loss_type, mask_type, training
):
    if not is_torch_1_2_plus:
        pytest.skip("Pytorch Version Under 1.2 is not supported for Enh task")

    if loss_type == "ci_sdr":
        inputs = torch.randn(2, 300)
        ilens = torch.LongTensor([300, 200])
        speech_refs = [torch.randn(2, 300).float(), torch.randn(2, 300).float()]
    else:
        # ci_sdr will fail if length is too short
        inputs = torch.randn(2, 100)
        ilens = torch.LongTensor([100, 80])
        speech_refs = [torch.randn(2, 100).float(), torch.randn(2, 100).float()]

    if loss_type not in ["si_snr", "ci_sdr"] and isinstance(encoder, ConvEncoder):
        with pytest.raises(TypeError):
            enh_model = ESPnetEnhancementModel(
                encoder=encoder,
                separator=separator,
                decoder=decoder,
                stft_consistency=stft_consistency,
                loss_type=loss_type,
                mask_type=mask_type,
            )
        return
    if stft_consistency and loss_type in ["mask_mse", "si_snr", "ci_sdr"]:
        with pytest.raises(ValueError):
            enh_model = ESPnetEnhancementModel(
                encoder=encoder,
                separator=separator,
                decoder=decoder,
                stft_consistency=stft_consistency,
                loss_type=loss_type,
                mask_type=mask_type,
            )
        return

    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        stft_consistency=stft_consistency,
        loss_type=loss_type,
        mask_type=mask_type,
    )

    if training:
        enh_model.train()
    else:
        enh_model.eval()

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(2)},
    }
    loss, stats, weight = enh_model(**kwargs)


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


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mask_type", ["IBM", "IRM", "IAM", "PSM", "PSM^2"])
@pytest.mark.parametrize(
    "loss_type", ["mask_mse", "magnitude", "spectrum", "spectrum_log"]
)
@pytest.mark.parametrize("num_spk", [1, 2, 3])
@pytest.mark.parametrize("use_noise_mask", [True, False])
@pytest.mark.parametrize("stft_consistency", [True, False])
def test_forward_with_beamformer_net(
    training, mask_type, loss_type, num_spk, use_noise_mask, stft_consistency
):
    if not is_torch_1_2_plus:
        pytest.skip("Pytorch Version Under 1.2 is not supported for Enh task")

    # Skip some testing cases
    if not loss_type.startswith("mask") and mask_type != "IBM":
        # `mask_type` has no effect when `loss_type` is not "mask..."
        return

    ch = 2
    inputs = random_speech[..., :ch].float()
    ilens = torch.LongTensor([16, 12])
    speech_refs = [torch.randn(2, 16, ch).float() for spk in range(num_spk)]
    noise_ref1 = torch.randn(2, 16, ch, dtype=torch.float)
    dereverb_ref1 = torch.randn(2, 16, ch, dtype=torch.float)
    encoder = STFTEncoder(n_fft=8, hop_length=2)
    decoder = STFTDecoder(n_fft=8, hop_length=2)

    if stft_consistency and loss_type in ["mask_mse", "si_snr", "ci_sdr"]:
        # skip this condition
        return

    beamformer = NeuralBeamformer(
        input_dim=5,
        loss_type=loss_type,
        num_spk=num_spk,
        use_wpe=True,
        wlayers=2,
        wunits=2,
        wprojs=2,
        use_dnn_mask_for_wpe=True,
        multi_source_wpe=True,
        use_beamformer=True,
        blayers=2,
        bunits=2,
        bprojs=2,
        badim=2,
        ref_channel=0,
        use_noise_mask=use_noise_mask,
        beamformer_type="mvdr_souden",
    )
    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        decoder=decoder,
        separator=beamformer,
        stft_consistency=stft_consistency,
        loss_type=loss_type,
        mask_type=mask_type,
    )
    if training:
        enh_model.train()
        if stft_consistency and not is_torch_1_2_plus:
            # torchaudio.functional.istft is only available with pytorch 1.2+
            return
    else:
        enh_model.eval()
        if not is_torch_1_2_plus:
            # torchaudio.functional.istft is only available with pytorch 1.2+
            return

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
        "noise_ref1": noise_ref1,
        "dereverb_ref1": dereverb_ref1,
    }
    loss, stats, weight = enh_model(**kwargs)
