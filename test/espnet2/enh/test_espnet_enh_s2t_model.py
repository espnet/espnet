import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.espnet_enh_s2t_model import ESPnetEnhS2TModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
from espnet2.enh.separator.rnn_separator import RNNSeparator


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
