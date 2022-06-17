import pytest
import torch
from packaging.version import parse as V

from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.null_decoder import NullDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.null_encoder import NullEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainL1, FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.multilayer_pit_solver import MultiLayerPITSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.separator.dc_crn_separator import DC_CRNSeparator
from espnet2.enh.separator.dccrn_separator import DCCRNSeparator
from espnet2.enh.separator.dprnn_separator import DPRNNSeparator
from espnet2.enh.separator.ineube_separator import iNeuBe
from espnet2.enh.separator.neural_beamformer import NeuralBeamformer
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.enh.separator.svoice_separator import SVoiceSeparator
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.separator.transformer_separator import TransformerSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


stft_encoder = STFTEncoder(
    n_fft=32,
    hop_length=16,
)

stft_encoder_bultin_complex = STFTEncoder(
    n_fft=32,
    hop_length=16,
    use_builtin_complex=True,
)

stft_decoder = STFTDecoder(
    n_fft=32,
    hop_length=16,
)

conv_encoder = ConvEncoder(
    channel=17,
    kernel_size=36,
    stride=18,
)

conv_decoder = ConvDecoder(
    channel=17,
    kernel_size=36,
    stride=18,
)

null_encoder = NullEncoder()

null_decoder = NullDecoder()

rnn_separator = RNNSeparator(
    input_dim=17,
    layer=1,
    unit=10,
)

dc_crn_separator = DC_CRNSeparator(input_dim=17, input_channels=[2, 2, 4])

dccrn_separator = DCCRNSeparator(input_dim=17, num_spk=1, kernel_num=[32, 64, 128])

dprnn_separator = DPRNNSeparator(input_dim=17, layer=1, unit=10, segment_size=4)

svoice_separator = SVoiceSeparator(
    input_dim=17,
    enc_dim=4,
    kernel_size=4,
    hidden_size=4,
    num_spk=2,
    num_layers=2,
    segment_size=4,
    bidirectional=False,
    input_normalize=False,
)

tcn_separator = TCNSeparator(
    input_dim=17,
    layer=2,
    stack=1,
    bottleneck_dim=10,
    hidden_dim=10,
    kernel=3,
)

transformer_separator = TransformerSeparator(
    input_dim=17,
    adim=8,
    aheads=2,
    layers=2,
    linear_units=10,
)


si_snr_loss = SISNRLoss()
tf_mse_loss = FrequencyDomainMSE()
tf_l1_loss = FrequencyDomainL1()
pit_wrapper = PITSolver(criterion=si_snr_loss)
multilayer_pit_solver = MultiLayerPITSolver(criterion=si_snr_loss)
fix_order_solver = FixedOrderSolver(criterion=tf_mse_loss)


@pytest.mark.parametrize(
    "encoder, decoder, separator", [(stft_encoder, stft_decoder, rnn_separator)]
)
@pytest.mark.parametrize("training", [True, False])
def test_criterion_behavior(encoder, decoder, separator, training):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    speech_refs = [torch.randn(2, 300).float(), torch.randn(2, 300).float()]
    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[PITSolver(criterion=SISNRLoss(only_for_test=True))],
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

    if training:
        with pytest.raises(AttributeError):
            loss, stats, weight = enh_model(**kwargs)
    else:
        loss, stats, weight = enh_model(**kwargs)


@pytest.mark.parametrize(
    "encoder, decoder",
    [
        (stft_encoder, stft_decoder),
        (stft_encoder_bultin_complex, stft_decoder),
        (conv_encoder, conv_decoder),
    ],
)
@pytest.mark.parametrize(
    "separator",
    [
        rnn_separator,
        dprnn_separator,
        dc_crn_separator,
        dccrn_separator,
        tcn_separator,
        transformer_separator,
    ],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss_wrappers", [[pit_wrapper, fix_order_solver]])
def test_single_channel_model(encoder, decoder, separator, training, loss_wrappers):
    if not isinstance(encoder, STFTEncoder) and isinstance(
        separator, (DCCRNSeparator, DC_CRNSeparator)
    ):
        # skip because DCCRNSeparator and DC_CRNSeparator only work
        # for complex spectrum features
        return
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    speech_refs = [torch.randn(2, 300).float(), torch.randn(2, 300).float()]
    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=loss_wrappers,
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


@pytest.mark.parametrize(
    "encoder, decoder",
    [
        (null_encoder, null_decoder),
    ],
)
@pytest.mark.parametrize(
    "separator",
    [
        svoice_separator,
    ],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss_wrappers", [[multilayer_pit_solver]])
def test_svoice_model(encoder, decoder, separator, training, loss_wrappers):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    speech_refs = [torch.randn(2, 300).float(), torch.randn(2, 300).float()]
    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=loss_wrappers,
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


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("n_mics", [1, 2])
@pytest.mark.parametrize("loss_wrappers", [[pit_wrapper]])
@pytest.mark.parametrize("output_from", ["dnn1", "dnn2"])
def test_ineube(n_mics, training, loss_wrappers, output_from):
    if not is_torch_1_9_plus:
        return
    inputs = torch.randn(1, 300, n_mics)
    ilens = torch.LongTensor([300])
    speech_refs = [torch.randn(1, 300).float(), torch.randn(1, 300).float()]
    from espnet2.enh.decoder.null_decoder import NullDecoder
    from espnet2.enh.encoder.null_encoder import NullEncoder

    encoder = NullEncoder()
    decoder = NullDecoder()
    separator = iNeuBe(
        2, mic_channels=n_mics, output_from=output_from, tcn_blocks=1, tcn_repeats=1
    )
    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=loss_wrappers,
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


pit_wrapper = PITSolver(criterion=FrequencyDomainMSE(compute_on_mask=True))


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mask_type", ["IBM", "IRM", "IAM", "PSM", "PSM^2"])
@pytest.mark.parametrize(
    "loss_type", ["mask_mse", "magnitude", "spectrum", "spectrum_log"]
)
@pytest.mark.parametrize("num_spk", [1, 2, 3])
@pytest.mark.parametrize("use_builtin_complex", [True, False])
@pytest.mark.parametrize("loss_wrappers", [[pit_wrapper]])
def test_forward_with_beamformer_net(
    training, mask_type, loss_type, num_spk, use_builtin_complex, loss_wrappers
):
    # Skip some testing cases
    if not loss_type.startswith("mask") and mask_type != "IBM":
        # `mask_type` has no effect when `loss_type` is not "mask..."
        return
    if not is_torch_1_9_plus and use_builtin_complex:
        # builtin complex support is only well supported in PyTorch 1.9+
        return

    ch = 3
    inputs = random_speech[..., :ch].float()
    ilens = torch.LongTensor([16, 12])
    speech_refs = [torch.randn(2, 16, dtype=torch.float) for spk in range(num_spk)]
    noise_ref1 = torch.randn(2, 16, ch, dtype=torch.float)
    dereverb_ref1 = torch.randn(2, 16, ch, dtype=torch.float)
    encoder = STFTEncoder(
        n_fft=8, hop_length=2, use_builtin_complex=use_builtin_complex
    )
    decoder = STFTDecoder(n_fft=8, hop_length=2)

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
        use_noise_mask=False,
        beamformer_type="mvdr_souden",
    )
    enh_model = ESPnetEnhancementModel(
        encoder=encoder,
        decoder=decoder,
        separator=beamformer,
        mask_module=None,
        loss_type=loss_type,
        mask_type=mask_type,
        loss_wrappers=loss_wrappers,
    )
    if training:
        enh_model.train()
    else:
        enh_model.eval()

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
        "dereverb_ref1": dereverb_ref1,
    }
    loss, stats, weight = enh_model(**kwargs)
    if mask_type in ("IBM", "IRM"):
        loss, stats, weight = enh_model(**kwargs, noise_ref1=noise_ref1)
