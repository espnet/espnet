from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.decoder.mega_decoder import MEGADecoder
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.rwkv_decoder import RWKVDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.espnet_transducer_model import ESPnetASRTransducerModel
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN


@pytest.fixture
def stats_file(tmp_path: Path):
    p = tmp_path / "stats.npy"

    count = 5
    x = np.random.randn(count, 10)
    s = x.sum(0)
    s = np.pad(s, [0, 1], mode="constant", constant_values=count)
    s2 = (x**2).sum(0)
    s2 = np.pad(s2, [0, 1], mode="constant", constant_values=0.0)

    stats = np.stack([s, s2])
    np.save(p, stats)

    return p


def prepare(model, input_size, vocab_size, batch_size, use_k2_modified_loss=False):
    n_token = vocab_size - 1

    label_len = [13, 9]

    # (b-flo): For k2 "modified", we need to ensure that T >= U after subsampling.
    if use_k2_modified_loss:
        feat_len = [i * 5 for i in label_len]
    else:
        feat_len = [15, 11]

    feats = torch.randn(batch_size, max(feat_len), input_size)
    labels = (torch.rand(batch_size, max(label_len)) * n_token % n_token).long()

    for i in range(2):
        feats[i, feat_len[i] :] = model.ignore_id
        labels[i, label_len[i] :] = model.ignore_id
    labels[labels == 0] = vocab_size - 2

    return feats, labels, torch.tensor(feat_len), torch.tensor(label_len)


def get_decoder(vocab_size, params):
    if "is_rwkv" in params:
        del params["is_rwkv"]

        decoder = RWKVDecoder(vocab_size, **params)
    elif "rnn_type" in params:
        decoder = RNNDecoder(vocab_size, **params)
    elif "block_size" in params:
        decoder = MEGADecoder(vocab_size, **params)
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
            {
                "auxiliary_ctc_weight": 0.1,
                "auxiliary_lm_loss_weight": 0.1,
                "normalize": "global",
            },
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
            {"specaug": True, "normalize": "utterance"},
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
                "num_left_chunks": 1,
            },
            {"embed_size": 4},
            {"joint_space_size": 4},
            {"transducer_weight": 1.0},
        ),
        (
            [
                {
                    "block_type": "branchformer",
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
                    "block_type": "branchformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                },
                {"block_type": "conv1d", "kernel_size": 1, "output_size": 2},
            ],
            {
                "dynamic_chunk_training": True,
                "short_chunk_size": 1,
                "num_left_chunks": 1,
            },
            {"embed_size": 4},
            {"joint_space_size": 4},
            {"transducer_weight": 1.0},
        ),
        (
            [
                {
                    "block_type": "ebranchformer",
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
                    "block_type": "ebranchformer",
                    "hidden_size": 4,
                    "linear_size": 4,
                    "conv_mod_kernel_size": 3,
                },
                {"block_type": "conv1d", "kernel_size": 1, "output_size": 2},
            ],
            {
                "dynamic_chunk_training": True,
                "short_chunk_size": 1,
                "num_left_chunks": 1,
            },
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
                "num_left_chunks": 1,
            },
            {"block_size": 4, "chunk_size": 3},
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
                }
            ],
            {},
            {"block_size": 4, "rel_pos_bias_type": "rotary"},
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
                },
                {"block_type": "conv1d", "kernel_size": 1, "output_size": 2},
            ],
            {
                "dynamic_chunk_training": True,
                "short_chunk_size": 1,
                "num_left_chunks": 1,
            },
            {"block_size": 4, "linear_size": 4, "is_rwkv": True},
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
                }
            ],
            {},
            {"block_size": 4, "linear_size": 4, "is_rwkv": True},
            {"joint_space_size": 4},
            {"report_cer": True, "report_wer": True},
        ),
    ],
)
def test_model_training(
    enc_params, enc_gen_params, dec_params, joint_net_params, main_params, stats_file
):
    batch_size = 2
    input_size = 10

    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    if dec_params.get("is_rwkv") is not None and not torch.cuda.is_available():
        pytest.skip("A GPU is required for WKV kernel computation")

    encoder = Encoder(input_size, enc_params, main_conf=enc_gen_params)
    decoder = get_decoder(vocab_size, dec_params)

    joint_network = JointNetwork(
        vocab_size, encoder.output_size, decoder.output_size, **joint_net_params
    )

    specaug = get_specaug() if main_params.pop("specaug", False) else None

    normalize = main_params.pop("normalize", None)
    if normalize is not None:
        if normalize == "utterance":
            normalize = UtteranceMVN(norm_means=True, norm_vars=True, eps=1e-13)
        else:
            normalize = GlobalMVN(stats_file, norm_means=True, norm_vars=True)

    model = ESPnetASRTransducerModel(
        vocab_size,
        token_list,
        frontend=None,
        specaug=specaug,
        normalize=normalize,
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


@pytest.mark.parametrize(
    "k2_params",
    [
        {},
        {"lm_scale": 0.25, "am_scale": 0.5},
        {"loss_type": "modified"},
    ],
)
def test_model_training_with_k2(k2_params):
    pytest.importorskip("k2")

    batch_size = 2
    input_size = 10

    token_list = ["<blank>", "a", "b", "c", "<space>"]
    vocab_size = len(token_list)

    encoder = Encoder(
        input_size,
        [
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "linear_size": 4,
                "conv_mod_kernel_size": 3,
            }
        ],
    )
    decoder = RNNDecoder(vocab_size, embed_size=8, hidden_size=8)

    joint_network = JointNetwork(
        vocab_size,
        encoder.output_size,
        decoder.output_size,
    )

    model = ESPnetASRTransducerModel(
        vocab_size,
        token_list,
        frontend=None,
        normalize=None,
        specaug=None,
        encoder=encoder,
        decoder=decoder,
        joint_network=joint_network,
        use_k2_pruned_loss=True,
        k2_pruned_loss_args=k2_params,
        report_cer=True,
        report_wer=True,
    )

    feats, labels, feat_len, label_len = prepare(
        model,
        input_size,
        vocab_size,
        batch_size,
        use_k2_modified_loss=True,
    )

    _ = model(feats, feat_len, labels, label_len)

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
