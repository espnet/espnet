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
        ([{"block_type": "rnn", "dim_hidden": 4}], "chainer", None),
        ([{"block_type": "rnn", "dim_hidden": 4}], "chainer_espnet1", None),
        (
            [{"block_type": "conv1d", "kernel_size": 1, "dim_output": 4}],
            "xavier_uniform",
            None,
        ),
        (
            [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 4}],
            "xavier_normal",
            None,
        ),
        (
            [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 4}],
            "kaiming_uniform",
            None,
        ),
        (
            [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 4}],
            "kaiming_normal",
            None,
        ),
        ([{"block_type": "rnn", "dim_hidden": 4}], "chainer", DefaultFrontend),
    ],
)
def test_model_initialization(body_conf, init_mode, frontend_class):
    token_list = ["<blank>", "a", "b", "c"]

    dim_vocab = len(token_list)
    dim_decoder = 4

    if frontend_class is not None:
        frontend = frontend_class()
        input_size = frontend.output_size()
    else:
        frontend = None
        input_size = 4

    encoder = Encoder(input_size, body_conf)

    decoder = RNNDecoder(dim_vocab, dim_embedding=4, dim_hidden=dim_decoder)
    joint_net = JointNetwork(dim_vocab, 4, dim_decoder, dim_joint_space=2)

    model = ESPnetASRTransducerModel(
        vocab_size=dim_vocab,
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

    dim_vocab = len(token_list)
    dim_encoder = 4
    dim_decoder = 4

    encoder = Encoder(4, [{"block_type": "rnn", "dim_hidden": 4}])
    decoder = RNNDecoder(dim_vocab, dim_embedding=dim_decoder, dim_hidden=dim_decoder)
    joint_net = JointNetwork(dim_vocab, dim_encoder, dim_decoder, dim_joint_space=2)

    model = ESPnetASRTransducerModel(
        vocab_size=dim_vocab,
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
