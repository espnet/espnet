import pytest
import torch

from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
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


def get_decoder(vocab_size, params):
    if "rnn_type" in params:
        decoder = RNNDecoder(vocab_size, **params)
    else:
        decoder = StatelessDecoder(vocab_size, **params)

    return decoder


def get_specaug():
    return SpecAug(
        apply_time_warp=True,
        apply_freq_mask=True,
        apply_time_mask=False,
    )


@pytest.mark.parametrize(
    "enc_params, enc_gen_params, dec_params, joint_net_params, main_params",
    [
        (
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                }
            ],
            {},
            {"rnn_type": "lstm", "num_layers": 2},
            {"joint_space_size": 4},
            {"report_cer": True, "report_wer": True},
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
            {},
            {"embed_size": 4},
            {"joint_space_size": 4},
            {"specaug": True},
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
            {},
            {"embed_size": 4},
            {"joint_space_size": 4},
            {"auxiliary_ctc_weight": 0.1, "auxiliary_lm_loss_weight": 0.1},
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
            {},
            {"embed_size": 4},
            {"joint_space_size": 4},
            {"transducer_weight": 1.0},
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
            {
                "dynamic_chunk_training": True,
                "short_chunk_size": 1,
                "left_chunk_size": 1,
            },
            {"embed_size": 4},
            {"joint_space_size": 4},
            {"transducer_weight": 1.0},
        ),
    ],
)
def test_model_training(
    enc_params, enc_gen_params, dec_params, joint_net_params, main_params
):
    batch_size = 2
    input_size = 10

    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder = Encoder(input_size, enc_params, main_conf=enc_gen_params)
    decoder = get_decoder(vocab_size, dec_params)

    joint_network = JointNetwork(
        vocab_size, encoder.output_size, decoder.output_size, **joint_net_params
    )

    specaug = get_specaug() if main_params.pop("specaug", False) else None

    model = ESPnetASRTransducerModel(
        vocab_size,
        token_list,
        frontend=None,
        specaug=specaug,
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

    if main_params.get("report_cer") or main_params.get("report_wer"):
        model.training = False

        _ = model(feats, feat_len, labels, label_len)


@pytest.mark.parametrize("extract_feats", [True, False])
def test_collect_feats(extract_feats):
    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder = Encoder(
        20,
        [
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "linear_size": 4,
                "conv_mod_kernel_size": 3,
            }
        ],
    )
    decoder = StatelessDecoder(vocab_size, embed_size=4)

    joint_network = JointNetwork(
        vocab_size, encoder.output_size, decoder.output_size, 8
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
    model.extract_feats_in_collect_stats = extract_feats

    feats_dict = model.collect_feats(
        torch.randn(2, 12),
        torch.tensor([12, 11]),
        torch.randn(2, 8),
        torch.tensor([8, 8]),
    )

    assert set(("feats", "feats_lengths")) == feats_dict.keys()
