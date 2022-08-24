import pytest
import torch

from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.error_calculator import ErrorCalculator
from espnet2.asr_transducer.joint_network import JointNetwork


@pytest.mark.parametrize(
    "report_opts, decoder_class, decoder_opts",
    [
        ({"report_cer": False, "report_wer": False}, RNNDecoder, {"hidden_size": 4}),
        ({"report_cer": True, "report_wer": True}, RNNDecoder, {"hidden_size": 4}),
        ({"report_cer": False, "report_wer": False}, StatelessDecoder, {}),
        ({"report_cer": True, "report_wer": True}, StatelessDecoder, {}),
    ],
)
def test_error_calculator_transducer(report_opts, decoder_class, decoder_opts):
    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder_size = 4

    decoder = decoder_class(vocab_size, embed_size=4, **decoder_opts)
    joint_net = JointNetwork(vocab_size, encoder_size, 4, joint_space_size=2)

    error_calc = ErrorCalculator(
        decoder,
        joint_net,
        token_list,
        "<space>",
        "<blank>",
        **report_opts,
    )

    enc_out = torch.randn(4, 30, encoder_size)
    target = torch.randint(0, vocab_size, [4, 20], dtype=torch.int32)

    with torch.no_grad():
        _, _ = error_calc(enc_out, target)
