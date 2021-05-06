# coding: utf-8

import argparse
import tempfile

import json
import pytest
import torch

from espnet.asr.pytorch_backend.asr_init import load_trained_model
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr_transducer import E2E
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.transducer.blocks import build_blocks


def make_train_args(**kwargs):
    train_defaults = dict(
        transformer_init="pytorch",
        etype="custom",
        custom_enc_input_layer="conv2d",
        custom_enc_self_attn_type="selfattn",
        custom_enc_positional_encoding_type="abs_pos",
        custom_enc_pw_activation_type="relu",
        custom_enc_conv_mod_activation_type="relu",
        enc_block_arch=[{"type": "transformer", "d_hidden": 2, "d_ff": 2, "heads": 1}],
        enc_block_repeat=1,
        dtype="custom",
        custom_dec_input_layer="embed",
        dec_block_arch=[{"type": "transformer", "d_hidden": 2, "d_ff": 2, "heads": 1}],
        dec_block_repeat=1,
        custom_dec_pw_activation_type="relu",
        dropout_rate_embed_decoder=0.0,
        joint_dim=2,
        joint_activation_type="tanh",
        aux_task_type=None,
        aux_task_weight=0.1,
        aux_task_layer_list=[],
        aux_ctc=False,
        aux_ctc_weight=1.0,
        aux_ctc_dropout_rate=0.0,
        trans_type="warp-transducer",
        rnnt_mode="rnnt_mode",
        char_list=["a", "e", "i", "o", "u"],
        sym_space="<space>",
        sym_blank="<blank>",
        report_cer=False,
        report_wer=False,
        search_type="default",
        score_norm_transducer=False,
        beam_size=1,
        nbest=1,
        verbose=0,
        outdir=None,
        rnnlm=None,
        model_module="espnet.nets.pytorch_backend.e2e_asr_transducer:E2E",
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def make_recog_args(**kwargs):
    recog_defaults = dict(
        batchsize=0,
        beam_size=1,
        nbest=1,
        verbose=0,
        search_type="default",
        nstep=1,
        max_sym_exp=2,
        u_max=5,
        prefix_alpha=2,
        score_norm_transducer=True,
        rnnlm=None,
        lm_weight=0.1,
    )
    recog_defaults.update(kwargs)

    return argparse.Namespace(**recog_defaults)


def get_default_scope_inputs():
    bs = 2
    idim = 12
    odim = 5

    ilens = [12, 4]
    olens = [5, 4]

    return bs, idim, odim, ilens, olens


def get_lm():
    n_layers = 1
    n_units = 4

    char_list = ["<blank>", "<space>", "a", "b", "c", "d", "<eos>"]

    rnnlm = lm_pytorch.ClassifierWithState(
        lm_pytorch.RNNLM(len(char_list), n_layers, n_units, typ="lstm")
    )

    return rnnlm


def get_wordlm():
    n_layers = 1
    n_units = 8

    char_list = ["<blank>", "<space>", "a", "b", "c", "d", "<eos>"]
    word_list = ["<blank>", "<unk>", "ab", "id", "ac", "bd", "<eos>"]

    char_dict = {x: i for i, x in enumerate(char_list)}
    word_dict = {x: i for i, x in enumerate(word_list)}

    word_rnnlm = lm_pytorch.ClassifierWithState(
        lm_pytorch.RNNLM(len(word_list), n_layers, n_units)
    )
    word_rnnlm = lm_pytorch.ClassifierWithState(
        extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor, word_dict, char_dict)
    )

    return word_rnnlm


def prepare(args):
    bs, idim, odim, ilens, olens = get_default_scope_inputs()
    n_token = odim - 1

    model = E2E(idim, odim, args)

    x = torch.randn(bs, max(ilens), idim)
    y = (torch.rand(bs, max(olens)) * n_token % n_token).long()

    for i in range(bs):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = model.ignore_id

    data = {}
    uttid_list = []
    for i in range(bs):
        data["utt%d" % i] = {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}],
        }
        uttid_list.append("utt%d" % i)

    return model, x, torch.tensor(ilens), y, data, uttid_list


@pytest.mark.parametrize(
    "train_dic, recog_dic",
    [
        ({}, {}),
        ({"enc_block_repeat": 2}, {}),
        ({"dec_block_repeat": 2}, {}),
        (
            {
                "enc_block_arch": [
                    {
                        "type": "conformer",
                        "d_hidden": 2,
                        "d_ff": 2,
                        "heads": 1,
                        "macaron_style": True,
                        "use_conv_mod": True,
                        "conv_mod_kernel": 1,
                    }
                ],
                "custom_enc_input_layer": "vgg2l",
                "custom_enc_self_attn_type": "rel_self_attn",
                "custom_enc_positional_encoding_type": "rel_pos",
            },
            {},
        ),
        (
            {
                "enc_block_arch": [
                    {
                        "type": "conformer",
                        "d_hidden": 2,
                        "d_ff": 2,
                        "heads": 2,
                        "macaron_style": False,
                        "use_conv_mod": True,
                        "conv_mod_kernel": 1,
                    }
                ],
            },
            {"custom_dec_pw_activation_type": "swish"},
        ),
        (
            {
                "enc_block_arch": [
                    {
                        "type": "tdnn",
                        "idim": 2,
                        "odim": 2,
                        "ctx_size": 2,
                        "dilation": 1,
                        "stride": 1,
                        "dropout-rate": 0.3,
                        "use-relu": True,
                        "use-batch-norm": True,
                    },
                    {
                        "type": "transformer",
                        "d_hidden": 2,
                        "d_ff": 2,
                        "heads": 1,
                        "dropout-rate": 0.3,
                        "att-dropout-rate": 0.2,
                        "pos-dropout-rate": 0.1,
                    },
                ],
            },
            {},
        ),
        (
            {
                "enc_block_arch": [
                    {
                        "type": "tdnn",
                        "idim": 2,
                        "odim": 2,
                        "ctx_size": 2,
                        "dilation": 1,
                        "stride": 1,
                        "dropout-rate": 0.3,
                        "use-relu": True,
                        "use-batch-norm": True,
                    },
                    {
                        "type": "conformer",
                        "d_hidden": 2,
                        "d_ff": 2,
                        "heads": 1,
                        "macaron_style": False,
                        "use_conv_mod": False,
                    },
                ],
                "custom_enc_input_layer": "linear",
                "custom_enc_self_attn_type": "rel_self_attn",
                "custom_enc_positional_encoding_type": "rel_pos",
            },
            {},
        ),
        (
            {
                "enc_block_arch": [
                    {
                        "type": "tdnn",
                        "idim": 2,
                        "odim": 2,
                        "ctx_size": 2,
                        "dilation": 1,
                        "stride": 1,
                    }
                ]
            },
            {},
        ),
        (
            {
                "dec_block_arch": [
                    {"type": "causal-conv1d", "idim": 2, "odim": 2, "kernel_size": 3},
                    {"type": "transformer", "d_hidden": 2, "d_ff": 2, "heads": 1},
                ]
            },
            {},
        ),
        ({"custom_enc_pw_activation_type": "swish"}, {}),
        ({"custom_enc_pw_activation_type": "hardtanh"}, {}),
        ({"custom_dec_pw_activation_type": "swish"}, {}),
        ({"custom_dec_pw_activation_type": "hardtanh"}, {}),
        ({"custom_enc_positional_encoding_type": "scaled_abs_pos"}, {}),
        ({"joint_activation_type": "relu"}, {}),
        ({"joint_activation_type": "swish"}, {}),
        ({"custom_enc_input_layer": "vgg2l"}, {}),
        ({"custom_enc_input_layer": "linear"}, {}),
        ({"report_cer": True, "report_wer": True}, {}),
        ({"report_cer": True, "beam_size": 2}, {}),
        ({}, {"beam_size": 2}),
        ({}, {"beam_size": 2, "nbest": 2, "score_norm_transducer": False}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 3, "prefix_alpha": 1}),
        ({}, {"beam_size": 2, "search_type": "tsd", "max_sym_exp": 3}),
        ({}, {"beam_size": 2, "search_type": "alsd"}),
        ({}, {"beam_size": 2, "search_type": "alsd", "u_max": 10}),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_wordlm()}),
    ],
)
def test_custom_transducer_trainable_and_decodable(train_dic, recog_dic):
    train_args = make_train_args(**train_dic)
    recog_args = make_recog_args(**recog_dic)

    model, x, ilens, y, data, uttid_list = prepare(train_args)

    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(x, ilens, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    beam_search = BeamSearchTransducer(
        decoder=model.decoder,
        joint_network=model.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
    )

    with torch.no_grad():
        nbest = model.recognize(x[0, : ilens[0]].numpy(), beam_search)

        print(y[0])
        print(nbest[0]["yseq"][1:-1])


def test_calculate_plot_attention():
    from espnet.nets.pytorch_backend.transformer import plot

    train_args = make_train_args(report_cer=True)
    model, x, ilens, y, data, uttid_list = prepare(train_args)

    attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
    plot.plot_multi_head_attention(data, uttid_list, attn_dict, "/tmp/espnet-test")


@pytest.mark.parametrize(
    "train_dic",
    [
        {
            "enc_block_repeat": 2,
            "aux_task_type": "default",
            "aux_task_layer_list": [0],
        },
        {
            "enc_block_arch": [
                {
                    "type": "conformer",
                    "d_hidden": 2,
                    "d_ff": 2,
                    "heads": 1,
                    "macaron_style": True,
                    "use_conv_mod": True,
                    "conv_mod_kernel": 1,
                }
            ],
            "custom_enc_input_layer": "vgg2l",
            "custom_enc_self_attn_type": "rel_self_attn",
            "custom_enc_positional_encoding_type": "rel_pos",
            "enc_block_repeat": 3,
            "aux_task_type": "symm_kl_div",
            "aux_task_layer_list": [0, 1],
        },
        {"aux_ctc": True, "aux_ctc_weight": 0.5},
        {"aux_cross_entropy": True, "aux_cross_entropy_weight": 0.5},
    ],
)
def test_auxiliary_task(train_dic):
    train_args = make_train_args(**train_dic)
    recog_args = make_recog_args()

    model, x, ilens, y, data, uttid_list = prepare(train_args)

    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(x, ilens, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    beam_search = BeamSearchTransducer(
        decoder=model.decoder,
        joint_network=model.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
    )

    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    torch.save(model.state_dict(), tmpdir + "/model.dummy.best")

    with open(tmpdir + "/model.json", "wb") as f:
        f.write(
            json.dumps(
                (12, 5, vars(train_args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )

    with torch.no_grad():
        model, _ = load_trained_model(tmpdir + "/model.dummy.best", training=False)

        nbest = model.recognize(x[0, : ilens[0]].numpy(), beam_search)

        print(y[0])
        print(nbest[0]["yseq"][1:-1])


def test_no_block_arch():
    _, idim, odim, _, _ = get_default_scope_inputs()
    args = make_train_args(enc_block_arch=None)

    with pytest.raises(ValueError):
        E2E(idim, odim, args)

    args = make_train_args(dec_block_arch=None)

    with pytest.raises(ValueError):
        E2E(idim, odim, args)


def test_invalid_input_layer_type():
    architecture = [
        {
            "type": "transformer",
            "d_hidden": 2,
            "d_ff": 2,
            "heads": 1,
        },
    ]

    with pytest.raises(NotImplementedError):
        _, _, _ = build_blocks("encoder", 4, "foo", architecture)


def test_invalid_architecture_layer_type():

    with pytest.raises(NotImplementedError):
        _, _, _ = build_blocks("encoder", 4, "linear", [{"type": "foo"}])


def test_invalid_block():
    with pytest.raises(ValueError):
        _, _, _ = build_blocks("encoder", 4, "linear", [{"foo": "foo"}])


def test_invalid_block_arguments():
    with pytest.raises(ValueError):
        _, _, _ = build_blocks("encoder", 4, "linear", [{"type": "transformer"}])

    with pytest.raises(ValueError):
        _, _, _ = build_blocks("encoder", 4, "linear", [{"type": "conformer"}])

    with pytest.raises(ValueError):
        _, _, _ = build_blocks(
            "encoder",
            4,
            "linear",
            [
                {
                    "type": "conformer",
                    "d_hidden": 4,
                    "d_ff": 8,
                    "heads": 1,
                    "macaron_style": False,
                    "use_conv_mod": True,
                }
            ],
        )

    with pytest.raises(ValueError):
        _, _, _ = build_blocks("decoder", 4, "embed", [{"type": "conformer"}])

    with pytest.raises(ValueError):
        _, _, _ = build_blocks("encoder", 4, "linear", [{"type": "tdnn"}])

    with pytest.raises(ValueError):
        _, _, _ = build_blocks("decoder", 4, "embed", [{"type": "causal-conv1d"}])

    with pytest.raises(ValueError):
        _, _, _ = build_blocks(
            "encoder",
            4,
            "embed",
            [
                {
                    "type": "transformer",
                    "d_hidden": 2,
                    "d_ff": 8,
                    "heads": 1,
                },
            ],
            positional_encoding_type="rel_pos",
            self_attn_type="self_attn",
        )


def test_invalid_block_io():
    with pytest.raises(ValueError):
        _, _, _ = build_blocks(
            "encoder",
            4,
            "linear",
            [
                {
                    "type": "transformer",
                    "d_hidden": 2,
                    "d_ff": 8,
                    "heads": 1,
                },
                {
                    "type": "transformer",
                    "d_hidden": 4,
                    "d_ff": 8,
                    "heads": 1,
                },
            ],
        )
