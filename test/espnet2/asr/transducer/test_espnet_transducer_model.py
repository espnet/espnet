import pytest
import torch

from espnet2.asr.transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr.transducer.encoder.encoder import Encoder
from espnet2.asr.transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr.transducer.joint_network import JointNetwork


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


def get_decoder(vocab_size, params):
    if "rnn_type" in params:
        decoder = RNNDecoder(vocab_size, **params)
    else:
        decoder = StatelessDecoder(vocab_size, **params)

    return decoder


@pytest.mark.parametrize(
    "enc_params, dec_params, joint_net_params, main_params",
    [
        (
            [{"block_type": "rnn", "dim_hidden": 4}],
            {"rnn_type": "lstm", "num_layers": 2},
            {"dim_joint_space": 4},
            {"report_cer": False, "report_wer": False},
        ),
        (
            [{"block_type": "rnn", "dim_hidden": 4}],
            {"dim_embedding": 4},
            {"dim_joint_space": 4},
            {},
        ),
        (
            [{"block_type": "rnn", "dim_hidden": 4}],
            {"dim_embedding": 4},
            {"dim_joint_space": 4},
            {"auxiliary_ctc_weight": 0.1, "auxiliary_lm_loss_weight": 0.1},
        ),
        (
            [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 4}],
            {"dim_embedding": 4},
            {"dim_joint_space": 4},
            {"auxiliary_ctc_weight": 0.1, "auxiliary_lm_loss_weight": 0.1},
        ),
    ],
)
def test_model_training(enc_params, dec_params, joint_net_params, main_params):
    batch_size = 2
    input_size = 10

    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder = Encoder(input_size, enc_params)
    decoder = get_decoder(vocab_size, dec_params)

    joint_network = JointNetwork(
        vocab_size, encoder.dim_output, decoder.dim_output, **joint_net_params
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
        **main_params,
    )

    feats, labels, feat_len, label_len = prepare(
        model, input_size, vocab_size, batch_size
    )

    _ = model(feats, feat_len, labels, label_len)
