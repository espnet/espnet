import pytest
import torch

from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.joint_network import JointNetwork


@pytest.mark.parametrize(
    "report_opts",
    [
        {"report_cer": False, "report_wer": False},
        {"report_cer": True, "report_wer": True},
    ],
)
def test_transducer_error_calculator(report_opts):
    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder_output_size = 4
    decoder_output_size = 4

    decoder = TransducerDecoder(
        vocab_size,
        hidden_size=decoder_output_size,
    )
    joint_net = JointNetwork(
        vocab_size, encoder_output_size, decoder_output_size, dim_joint_space=2
    )

    error_calc = ErrorCalculatorTransducer(
        decoder,
        joint_net,
        token_list,
        "<space>",
        "<blank>",
        **report_opts,
    )

    enc_out = torch.randn(4, 30, encoder_output_size)
    target = torch.randint(0, vocab_size, [4, 20], dtype=torch.int32)

    with torch.no_grad():
        _, _ = error_calc(enc_out, target)
