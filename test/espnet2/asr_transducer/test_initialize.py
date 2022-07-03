import pytest

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr_transducer.initialize import initialize
from espnet2.asr_transducer.joint_network import JointNetwork


@pytest.mark.parametrize(
    "body_conf, init_mode, frontend_class",
    [
        (
            [{"block_type": "conv1d", "kernel_size": 1, "output_size": 4}],
            "xavier_uniform",
            None,
        ),
        (
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                }
            ],
            "xavier_normal",
            None,
        ),
        (
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                }
            ],
            "kaiming_uniform",
            None,
        ),
        (
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                },
                {"block_type": "conv1d", "kernel_size": 1, "output_size": 2},
            ],
            "kaiming_normal",
            None,
        ),
        (
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                }
            ],
            "chainer",
            DefaultFrontend,
        ),
    ],
)
def test_model_initialization(body_conf, init_mode, frontend_class):
    token_list = ["<blank>", "a", "b", "c"]

    vocab_size = len(token_list)
    decoder_size = 4

    if frontend_class is not None:
        frontend = frontend_class()
        input_size = frontend.output_size()
    else:
        frontend = None
        input_size = 8

    encoder = Encoder(input_size, body_conf)

    decoder = RNNDecoder(vocab_size, embed_size=4, hidden_size=decoder_size)
    joint_net = JointNetwork(vocab_size, 4, decoder_size, joint_space_size=2)

    model = ESPnetASRTransducerModel(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=frontend,
        specaug=None,
        normalize=None,
        encoder=encoder,
        decoder=decoder,
        joint_network=joint_net,
    )

    initialize(model, init_mode)


def test_wrong_model_initialization():
    token_list = ["<blank>", "a", "b", "c"]

    vocab_size = len(token_list)
    decoder_size = 4

    encoder = Encoder(
        8,
        [
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "linear_size": 4,
                "conv_mod_kernel_size": 3,
            }
        ],
    )
    decoder = RNNDecoder(vocab_size, embed_size=decoder_size, hidden_size=decoder_size)
    joint_net = JointNetwork(
        vocab_size, encoder.output_size, decoder.output_size, joint_space_size=2
    )

    model = ESPnetASRTransducerModel(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        encoder=encoder,
        decoder=decoder,
        joint_network=joint_net,
    )

    with pytest.raises(ValueError):
        initialize(model, "foo")
