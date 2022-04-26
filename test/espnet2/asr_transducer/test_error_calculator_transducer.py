import pytest
import torch

from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.error_calculator import ErrorCalculator
from espnet2.asr_transducer.joint_network import JointNetwork


@pytest.mark.parametrize(
    "report_opts, decoder_class, decoder_opts",
    [
        ({"report_cer": False, "report_wer": False}, RNNDecoder, {"dim_hidden": 4}),
        ({"report_cer": True, "report_wer": True}, RNNDecoder, {"dim_hidden": 4}),
        ({"report_cer": False, "report_wer": False}, StatelessDecoder, {}),
        ({"report_cer": True, "report_wer": True}, StatelessDecoder, {}),
    ],
)
def test_error_calculator(report_opts, decoder_class, decoder_opts):
    token_list = ["<blank>", "a", "b", "c", "<space>"]
    dim_vocab = len(token_list)

    dim_encoder = 4

    decoder = decoder_class(dim_vocab, dim_embedding=4, **decoder_opts)
    joint_net = JointNetwork(dim_vocab, dim_encoder, 4, dim_joint_space=2)

    error_calc = ErrorCalculator(
        decoder,
        joint_net,
        token_list,
        "<space>",
        "<blank>",
        **report_opts,
    )

    enc_out = torch.randn(4, 30, dim_encoder)
    target = torch.randint(0, dim_vocab, [4, 20], dtype=torch.int32)

    with torch.no_grad():
        _, _ = error_calc(enc_out, target)
