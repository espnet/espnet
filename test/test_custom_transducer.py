# coding: utf-8

import argparse
import json
import tempfile

import pytest
import torch
from packaging.version import parse as V

import espnet.lm.pytorch_backend.extlm as extlm_pytorch
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr_transducer import E2E
from espnet.nets.pytorch_backend.transducer.blocks import build_blocks

is_torch_1_5_plus = V(torch.__version__) >= V("1.5.0")


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
        transducer_loss_weight=1.0,
        use_ctc_loss=False,
        ctc_loss_weight=0.0,
        ctc_loss_dropout_rate=0.0,
        use_lm_loss=False,
        lm_loss_weight=0.0,
        use_aux_transducer_loss=False,
        aux_transducer_loss_weight=0.0,
        aux_transducer_loss_enc_output_layers=[],
        use_symm_kl_div_loss=False,
        symm_kl_div_loss_weight=0.0,
        char_list=["a", "e", "i", "o", "u"],
        sym_space="<space>",
        sym_blank="<blank>",
        report_cer=False,
        report_wer=False,
        search_type="default",
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
        softmax_temperature=1.0,
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

    ilens = [15, 11]
    olens = [13, 9]

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

    feats = torch.randn(bs, max(ilens), idim)
    labels = (torch.rand(bs, max(olens)) * n_token % n_token).long()

    for i in range(bs):
        feats[i, ilens[i] :] = -1
        labels[i, olens[i] :] = model.ignore_id

    data = {}
    uttid_list = []
    for i in range(bs):
        data["utt%d" % i] = {
            "input": [{"shape": [ilens[i], idim]}],
            "output": [{"shape": [olens[i]]}],
        }
        uttid_list.append("utt%d" % i)

    return model, feats, torch.tensor(ilens), labels, data, uttid_list


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
                "custom_enc_input_layer": "linear",
                "custom_enc_positional_encoding_type": "abs_pos",
                "enc_block_arch": [
                    {
                        "type": "transformer",
                        "d_hidden": 32,
                        "d_ff": 4,
                        "heads": 1,
                    },
                    {
                        "type": "conv1d",
                        "idim": 32,
                        "odim": 16,
                        "kernel_size": 3,
                        "dilation": 2,
                        "stride": 2,
                        "dropout-rate": 0.3,
                        "use-relu": True,
                        "use-batch-norm": True,
                    },
                    {
                        "type": "transformer",
                        "d_hidden": 16,
                        "d_ff": 4,
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
                        "type": "conv1d",
                        "idim": 8,
                        "odim": 8,
                        "kernel_size": 2,
                        "dilation": 2,
                        "stride": 1,
                        "dropout-rate": 0.3,
                        "use-relu": True,
                        "use-batch-norm": True,
                    },
                    {
                        "type": "conformer",
                        "d_hidden": 8,
                        "d_ff": 4,
                        "heads": 1,
                        "macaron_style": False,
                        "use_conv_mod": False,
                    },
                ],
                "custom_enc_self_attn_type": "rel_self_attn",
                "custom_enc_positional_encoding_type": "rel_pos",
            },
            {},
        ),
        (
            {
                "enc_block_arch": [
                    {
                        "type": "conv1d",
                        "idim": 2,
                        "odim": 2,
                        "kernel_size": 2,
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
                    {
                        "type": "causal-conv1d",
                        "idim": 8,
                        "odim": 8,
                        "kernel_size": 3,
                        "dropout-rate": 0.3,
                        "use-relu": True,
                        "use-batch-norm": True,
                    },
                    {"type": "transformer", "d_hidden": 8, "d_ff": 4, "heads": 1},
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
        ({}, {"beam_size": 2, "search_type": "maes", "nstep": 3, "prefix_alpha": 1}),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_wordlm()}),
        ({}, {"beam_size": 2, "search_type": "maes", "nstep": 4, "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "maes", "rnnlm": get_wordlm()}),
        ({}, {"beam_size": 2, "softmax_temperature": 2.0, "rnnlm": get_wordlm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "softmax_temperature": 5.0}),
    ],
)
def test_custom_transducer_trainable_and_decodable(train_dic, recog_dic):
    train_args = make_train_args(**train_dic)
    recog_args = make_recog_args(**recog_dic)

    model, feats, feats_len, labels, data, uttid_list = prepare(train_args)

    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(feats, feats_len, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()

    beam_search = BeamSearchTransducer(
        decoder=model.decoder,
        joint_network=model.transducer_tasks.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
        softmax_temperature=recog_args.softmax_temperature,
    )

    with torch.no_grad():
        nbest = model.recognize(feats[0, : feats_len[0]].numpy(), beam_search)

        print(nbest[0]["yseq"][1:-1])


@pytest.mark.execution_timeout(4)
def test_calculate_plot_attention():
    from espnet.nets.pytorch_backend.transformer import plot

    train_args = make_train_args(report_cer=True)
    model, feats, feats_len, labels, data, uttid_list = prepare(train_args)

    model.attention_plot_class
    attn_dict = model.calculate_all_attentions(feats[0:1], feats_len[0:1], labels[0:1])

    plot.plot_multi_head_attention(data, uttid_list, attn_dict, "/tmp/espnet-test")


@pytest.mark.parametrize(
    "train_dic",
    [
        {
            "enc_block_repeat": 2,
            "use_aux_transducer_loss": True,
            "aux_transducer_loss_enc_output_layers": [0],
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
            "use_aux_transducer_loss": True,
            "aux_transducer_loss_enc_output_layers": [0, 1],
        },
        {"aux_ctc": True, "aux_ctc_weight": 0.5},
        {"aux_cross_entropy": True, "aux_cross_entropy_weight": 0.5},
    ],
)
def test_auxiliary_task(train_dic):
    train_args = make_train_args(**train_dic)
    recog_args = make_recog_args()

    model, feats, feats_len, labels, data, uttid_list = prepare(train_args)

    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(feats, feats_len, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()

    beam_search = BeamSearchTransducer(
        decoder=model.decoder,
        joint_network=model.transducer_tasks.joint_network,
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

        nbest = model.recognize(feats[0, : feats_len[0]].numpy(), beam_search)

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
        _, _, _ = build_blocks("encoder", 4, "embed", [{"type": "causal-conv1d"}])

    with pytest.raises(ValueError):
        _, _, _ = build_blocks("decoder", 4, "embed", [{"type": "conv1d"}])

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


@pytest.mark.parametrize(
    "train_dic",
    [
        {},
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
        {
            "enc_block_arch": [
                {
                    "type": "conv1d",
                    "idim": 2,
                    "odim": 2,
                    "kernel_size": 2,
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
                    "macaron_style": False,
                    "use_conv_mod": False,
                },
            ],
            "custom_enc_input_layer": "linear",
        },
        {
            "dec_block_arch": [
                {"type": "causal-conv1d", "idim": 2, "odim": 2, "kernel_size": 1},
                {"type": "transformer", "d_hidden": 2, "d_ff": 2, "heads": 1},
            ]
        },
    ],
)
@pytest.mark.parametrize(
    "recog_dic",
    [
        {},
        {"beam_size": 2, "search_type": "default"},
        {"beam_size": 2, "search_type": "alsd"},
        {"beam_size": 2, "search_type": "tsd"},
        {"beam_size": 2, "search_type": "nsc"},
        {"beam_size": 2, "search_type": "maes"},
    ],
)
@pytest.mark.parametrize(
    "quantize_dic",
    [
        {"mod": {torch.nn.Linear}, "dtype": torch.qint8},
        {"mod": {torch.nn.Linear}, "dtype": torch.float16},
        {"mod": {torch.nn.LSTM}, "dtype": torch.qint8},
        {"mod": {torch.nn.LSTM}, "dtype": torch.float16},
        {"mod": {torch.nn.Linear, torch.nn.LSTM}, "dtype": torch.qint8},
        {"mod": {torch.nn.Linear, torch.nn.LSTM}, "dtype": torch.float16},
    ],
)
@pytest.mark.execution_timeout(4)
def test_dynamic_quantization(train_dic, recog_dic, quantize_dic):
    train_args = make_train_args(**train_dic)
    recog_args = make_recog_args(**recog_dic)

    model, feats, feats_len, _, _, _ = prepare(train_args)

    if not is_torch_1_5_plus and (
        torch.nn.Linear in quantize_dic["mod"]
        and quantize_dic["dtype"] == torch.float16
    ):
        # In recognize(...) from asr.py we raise ValueError however
        # AssertionError is originaly raised by torch.
        with pytest.raises(AssertionError):
            model = torch.quantization.quantize_dynamic(
                model,
                quantize_dic["mod"],
                dtype=quantize_dic["dtype"],
            )
        pytest.skip("Skip rest of the test after checking AssertionError")
    else:
        model = torch.quantization.quantize_dynamic(
            model,
            quantize_dic["mod"],
            dtype=quantize_dic["dtype"],
        )

    beam_search = BeamSearchTransducer(
        decoder=model.decoder,
        joint_network=model.transducer_tasks.joint_network,
        beam_size=recog_args.beam_size,
        lm=recog_args.rnnlm,
        lm_weight=recog_args.lm_weight,
        search_type=recog_args.search_type,
        max_sym_exp=recog_args.max_sym_exp,
        u_max=recog_args.u_max,
        nstep=recog_args.nstep,
        prefix_alpha=recog_args.prefix_alpha,
        score_norm=recog_args.score_norm_transducer,
        quantization=True,
    )

    with torch.no_grad():
        model.recognize(feats[0, : feats_len[0]].numpy(), beam_search)


@pytest.mark.parametrize(
    "train_dic, subsample",
    [
        ({}, 4),
        ({"custom_enc_input_layer": "vgg2l"}, 4),
        ({"custom_enc_input_layer": "linear"}, 1),
    ],
)
def test_subsampling(train_dic, subsample):
    train_args = make_train_args(**train_dic)

    model, feats, feats_len, _, _, _ = prepare(train_args)

    assert model.get_total_subsampling_factor() == subsample
