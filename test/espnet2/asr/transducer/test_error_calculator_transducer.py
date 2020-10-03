import torch

from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer


def test_error_calculator_transducer():
    x = torch.randn(2, 9, 20)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 6, [2, 4], dtype=torch.long)

    sym_blank = "<blank>"
    sym_space = "<space>"
    token_list = [sym_blank] + ["a", "b", "c", "d"] + [sym_space]

    encoder = RNNEncoder(20, hidden_size=24, output_size=12)
    encoder_out, _, _ = encoder(x, x_lens)

    decoder = RNNDecoder(
        6,
        12,
        hidden_size=12,
        embed_pad=0,
        rnn_type="lstm",
        use_attention=False,
        use_output=False,
    )
    joint_net = JointNetwork(6, 12, decoder.dunits, joint_space_size=20)

    error_calculator = ErrorCalculatorTransducer(
        decoder,
        joint_net,
        token_list,
        sym_space,
        sym_blank,
        report_cer=True,
        report_wer=True,
    )

    cer, wer = error_calculator(encoder_out, t)
