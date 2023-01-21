import pytest
import torch
from packaging.version import parse as V

from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_model_tse import ESPnetExtractionModel
from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainMSE
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


stft_encoder = STFTEncoder(n_fft=32, hop_length=16)
stft_encoder_bultin_complex = STFTEncoder(
    n_fft=32, hop_length=16, use_builtin_complex=True
)
stft_decoder = STFTDecoder(n_fft=32, hop_length=16)
conv_encoder = ConvEncoder(channel=17, kernel_size=36, stride=18)
conv_decoder = ConvDecoder(channel=17, kernel_size=36, stride=18)

td_speakerbeam_extractor = TDSpeakerBeamExtractor(
    input_dim=17,
    layer=3,
    stack=2,
    bottleneck_dim=8,
    hidden_dim=16,
    skip_dim=8,
    i_adapt_layer=3,
    adapt_enroll_dim=8,
)

si_snr_loss = SISNRLoss()
tf_mse_loss = FrequencyDomainMSE()

pit_wrapper = PITSolver(criterion=si_snr_loss)
fix_order_solver = FixedOrderSolver(criterion=tf_mse_loss)


@pytest.mark.parametrize(
    "encoder, decoder",
    [
        (stft_encoder, stft_decoder),
        (stft_encoder_bultin_complex, stft_decoder),
        (conv_encoder, conv_decoder),
    ],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("loss_wrapper", [pit_wrapper, fix_order_solver])
def test_criterion_behavior_validation(
    encoder, decoder, training, num_spk, loss_wrapper
):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    speech_refs = [torch.randn(2, 300).float(), torch.randn(2, 300).float()]
    enroll_refs = [torch.randn(2, 400).float(), torch.randn(2, 400).float()]
    aux_lens = [torch.LongTensor([400, 300]), torch.LongTensor([400, 350])]
    if num_spk == 1:
        speech_refs = speech_refs[:1]
        enroll_refs = enroll_refs[:1]
        aux_lens = aux_lens[:1]
    enh_model = ESPnetExtractionModel(
        encoder=encoder,
        extractor=td_speakerbeam_extractor,
        decoder=decoder,
        loss_wrappers=[loss_wrapper],
        num_spk=num_spk,
        share_encoder=True,
    )

    if training:
        enh_model.train()
    else:
        enh_model.eval()

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
        **{"enroll_ref{}".format(i + 1): enroll_refs[i] for i in range(num_spk)},
        **{"enroll_ref{}_lengths".format(i + 1): aux_lens[i] for i in range(num_spk)},
    }

    if training:
        loss, stats, weight = enh_model(**kwargs)
    else:
        loss, stats, weight = enh_model(**kwargs)


@pytest.mark.parametrize("encoder, decoder", [(conv_encoder, conv_decoder)])
@pytest.mark.parametrize("extractor", [td_speakerbeam_extractor])
def test_criterion_behavior_noise_dereverb(encoder, decoder, extractor):
    with pytest.raises(ValueError):
        ESPnetExtractionModel(
            encoder=encoder,
            extractor=extractor,
            decoder=decoder,
            loss_wrappers=[PITSolver(criterion=SISNRLoss(is_noise_loss=True))],
        )
    with pytest.raises(ValueError):
        ESPnetExtractionModel(
            encoder=encoder,
            extractor=extractor,
            decoder=decoder,
            loss_wrappers=[PITSolver(criterion=SISNRLoss(is_dereverb_loss=True))],
        )
