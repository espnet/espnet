import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.diar.decoder.linear_decoder import LinearDecoder
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.diar.layers.multi_mask import MultiMask
from espnet2.diar.separator.tcn_separator_nomask import TCNSeparatorNomask
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_enh_s2t_model import ESPnetEnhS2TModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.separator.rnn_separator import RNNSeparator
from espnet2.layers.label_aggregation import LabelAggregate


enh_stft_encoder = STFTEncoder(
    n_fft=32,
    hop_length=16,
)

enh_stft_decoder = STFTDecoder(
    n_fft=32,
    hop_length=16,
)

enh_rnn_separator = RNNSeparator(
    input_dim=17,
    layer=1,
    unit=10,
    num_spk=1,
)

si_snr_loss = SISNRLoss()

fix_order_solver = FixedOrderSolver(criterion=si_snr_loss)

default_frontend = DefaultFrontend(
    fs=300,
    n_fft=32,
    win_length=32,
    hop_length=24,
    n_mels=32,
)

token_list = ["<blank>", "<space>", "a", "e", "i", "o", "u", "<sos/eos>"]

asr_transformer_encoder = TransformerEncoder(
    32,
    output_size=16,
    linear_units=16,
    num_blocks=2,
)

asr_transformer_decoder = TransformerDecoder(
    len(token_list),
    16,
    linear_units=16,
    num_blocks=2,
)

asr_ctc = CTC(odim=len(token_list), encoder_output_size=16)


@pytest.mark.parametrize(
    "enh_encoder, enh_decoder",
    [(enh_stft_encoder, enh_stft_decoder)],
)
@pytest.mark.parametrize("enh_separator", [enh_rnn_separator])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss_wrappers", [[fix_order_solver]])
@pytest.mark.parametrize("frontend", [default_frontend])
@pytest.mark.parametrize("s2t_encoder", [asr_transformer_encoder])
@pytest.mark.parametrize("s2t_decoder", [asr_transformer_decoder])
@pytest.mark.parametrize("s2t_ctc", [asr_ctc])
def test_enh_asr_model(
    enh_encoder,
    enh_decoder,
    enh_separator,
    training,
    loss_wrappers,
    frontend,
    s2t_encoder,
    s2t_decoder,
    s2t_ctc,
):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    speech_ref = torch.randn(2, 300).float()
    text = torch.LongTensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    text_lengths = torch.LongTensor([5, 5])
    enh_model = ESPnetEnhancementModel(
        encoder=enh_encoder,
        separator=enh_separator,
        decoder=enh_decoder,
        mask_module=None,
        loss_wrappers=loss_wrappers,
    )
    s2t_model = ESPnetASRModel(
        vocab_size=len(token_list),
        token_list=token_list,
        frontend=frontend,
        encoder=s2t_encoder,
        decoder=s2t_decoder,
        ctc=s2t_ctc,
        specaug=None,
        normalize=None,
        preencoder=None,
        postencoder=None,
        joint_network=None,
    )
    enh_s2t_model = ESPnetEnhS2TModel(
        enh_model=enh_model,
        s2t_model=s2t_model,
    )

    if training:
        enh_s2t_model.train()
    else:
        enh_s2t_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "speech_ref1": speech_ref,
        "text": text,
        "text_lengths": text_lengths,
    }
    loss, stats, weight = enh_s2t_model(**kwargs)


label_aggregator = LabelAggregate(
    win_length=32,
    hop_length=16,
)

enh_encoder = ConvEncoder(
    channel=17,
    kernel_size=32,
    stride=16,
)

enh_decoder = ConvDecoder(
    channel=17,
    kernel_size=32,
    stride=16,
)

tcn_separator = TCNSeparatorNomask(
    input_dim=enh_encoder.output_dim,
    layer=2,
    stack=1,
    bottleneck_dim=10,
    hidden_dim=10,
    kernel=3,
)

mask_module = MultiMask(
    bottleneck_dim=10,
    max_num_spk=3,
    input_dim=enh_encoder.output_dim,
)

diar_frontend = DefaultFrontend(
    n_fft=32,
    win_length=32,
    hop_length=16,
    n_mels=32,
)

diar_encoder = TransformerEncoder(
    input_layer="linear",
    num_blocks=1,
    linear_units=32,
    output_size=16,
    attention_heads=2,
    input_size=tcn_separator.output_dim + diar_frontend.output_size(),
)

diar_decoder = LinearDecoder(
    num_spk=2,
    encoder_output_size=diar_encoder.output_size(),
)


@pytest.mark.parametrize("label_aggregator", [label_aggregator])
@pytest.mark.parametrize("enh_encoder, enh_decoder", [(enh_encoder, enh_decoder)])
@pytest.mark.parametrize("enh_separator", [tcn_separator])
@pytest.mark.parametrize("mask_module", [mask_module])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss_wrappers", [[fix_order_solver]])
@pytest.mark.parametrize("diar_frontend", [diar_frontend])
@pytest.mark.parametrize("diar_encoder, diar_decoder", [(diar_encoder, diar_decoder)])
def test_enh_diar_model(
    enh_encoder,
    enh_decoder,
    enh_separator,
    mask_module,
    training,
    loss_wrappers,
    diar_frontend,
    diar_encoder,
    diar_decoder,
    label_aggregator,
):
    inputs = torch.randn(2, 300)
    speech_ref = torch.randn(2, 300).float()
    text = torch.randint(high=2, size=(2, 300, 2))
    enh_model = ESPnetEnhancementModel(
        encoder=enh_encoder,
        separator=enh_separator,
        decoder=enh_decoder,
        mask_module=mask_module,
        loss_wrappers=loss_wrappers,
    )
    diar_model = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        frontend=diar_frontend,
        encoder=diar_encoder,
        decoder=diar_decoder,
        specaug=None,
        normalize=None,
        attractor=None,
    )
    enh_s2t_model = ESPnetEnhS2TModel(
        enh_model=enh_model,
        s2t_model=diar_model,
    )

    if training:
        enh_s2t_model.train()
    else:
        enh_s2t_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_ref1": speech_ref,
        "speech_ref2": speech_ref,
        "text": text,
    }
    loss, stats, weight = enh_s2t_model(**kwargs)
