import pytest
import torch

from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr_transducer.joint_network import JointNetwork


def prepare(model, input_size, vocab_size, batch_size):
    n_token = vocab_size - 1

    feat_len = [15, 11]
    label_len = [13, 9]

    feats = torch.randn(batch_size, max(feat_len), input_size)
    labels = (torch.rand(batch_size, max(label_len)) * n_token % n_token).long()

    for i in range(2):
        feats[i, feat_len[i] :] = model.ignore_id
        labels[i, label_len[i] :] = model.ignore_id
    labels[labels == 0] = vocab_size - 2

    return feats, labels, torch.tensor(feat_len), torch.tensor(label_len)


@pytest.mark.parametrize(
    "act_type, act_params",
    [
        ("ftswish", {"ftswish_threshold": -0.25, "ftswish_mean_shift": -0.1}),
        ("hardtanh", {"hardtanh_min_val": -2, "hardtanh_max_val": 2}),
        ("leaky_relu", {"leakyrelu_neg_slope": 0.02}),
        ("mish", {"softplus_beta": 1.125, "softplus_threshold": 10}),
        ("relu", {}),
        ("selu", {}),
        ("smish", {"smish_alpha": 1.125, "smish_beta": 1.125}),
        ("swish", {}),
        ("swish", {"swish_beta": 1.125}),
        ("tanh", {}),
    ],
)
def test_activation(act_type, act_params):
    batch_size = 2
    input_size = 10

    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder = Encoder(
        input_size,
        [
            {
                "block_type": "conformer",
                "hidden_size": 8,
                "linear_size": 4,
                "conv_mod_kernel_size": 3,
            }
        ],
        main_conf=act_params,
    )
    decoder = StatelessDecoder(vocab_size, embed_size=4)

    joint_network = JointNetwork(
        vocab_size,
        encoder.output_size,
        decoder.output_size,
        joint_activation_type=act_type,
        **act_params,
    )

    model = ESPnetASRTransducerModel(
        vocab_size,
        token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        encoder=encoder,
        decoder=decoder,
        joint_network=joint_network,
    )

    feats, labels, feat_len, label_len = prepare(
        model, input_size, vocab_size, batch_size
    )

    _ = model(feats, feat_len, labels, label_len)
