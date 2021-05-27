# coding: utf-8

import argparse
import tempfile

import json
import numpy as np
import pytest
import torch

from espnet.asr.pytorch_backend.asr_init import load_trained_model
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.e2e_asr_transducer import E2E
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.nets_utils import pad_list


def get_default_train_args(**kwargs):
    train_defaults = dict(
        etype="vggblstmp",
        elayers=1,
        subsample="1_2_2_1_1",
        eunits=4,
        eprojs=4,
        dtype="lstm",
        dlayers=1,
        dunits=4,
        dec_embed_dim=4,
        atype="location",
        adim=4,
        aheads=2,
        awin=2,
        aconv_chans=2,
        aconv_filts=5,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_embed_decoder=0.0,
        joint_dim=2,
        joint_activation_type="tanh",
        transducer_weight=1.0,
        aux_task_type=None,
        aux_task_weight=0.1,
        aux_task_layer_list=[],
        aux_ctc=False,
        aux_ctc_weight=1.0,
        aux_ctc_dropout_rate=0.0,
        use_frontend=False,
        trans_type="warp-transducer",
        char_list=["a", "b", "c", "d"],
        sym_space="<space>",
        sym_blank="<blank>",
        report_cer=False,
        report_wer=False,
        score_norm_transducer=True,
        beam_size=1,
        nbest=1,
        verbose=0,
        outdir=None,
        rnnlm=None,
        model_module="espnet.nets.pytorch_backend.e2e_asr_transducer:E2E",
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def get_default_recog_args(**kwargs):
    recog_defaults = dict(
        batchsize=0,
        beam_size=1,
        nbest=1,
        verbose=0,
        search_type="default",
        nstep=1,
        max_sym_exp=2,
        prefix_alpha=2,
        u_max=5,
        score_norm_transducer=True,
        rnnlm=None,
        lm_weight=0.1,
    )
    recog_defaults.update(kwargs)

    return argparse.Namespace(**recog_defaults)


def get_default_scope_inputs():
    idim = 15
    odim = 4

    ilens = [12, 8]
    olens = [8, 4]

    return idim, odim, ilens, olens


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


def prepare_inputs(idim, odim, ilens, olens, is_cuda=False):
    np.random.seed(1)

    xs = [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
    ilens = torch.from_numpy(ilens).long()

    if is_cuda:
        xs_pad = xs_pad.cuda()
        ys_pad = ys_pad.cuda()
        ilens = ilens.cuda()

    return xs_pad, ilens, ys_pad


@pytest.mark.parametrize(
    "train_dic, recog_dic",
    [
        ({}, {}),
        ({"eprojs": 4}, {}),
        ({"dlayers": 2}, {}),
        ({"etype": "gru"}, {}),
        ({"etype": "blstm"}, {}),
        ({"etype": "blstmp", "elayers": 2, "eprojs": 4}, {}),
        ({"etype": "vgggru"}, {}),
        ({"etype": "vggbru"}, {}),
        ({"etype": "vgggrup", "elayers": 2, "eprojs": 4}, {}),
        ({"dtype": "gru"}, {}),
        ({"dtype": "bgrup"}, {}),
        ({"dtype": "gru", "dlayers": 2}, {}),
        ({"joint-activation-type": "relu"}, {}),
        ({"joint-activation-type": "swish"}, {}),
        ({}, {"score_norm_transducer": False}),
        ({"report_cer": True, "report_wer": True}, {}),
        ({}, {"nbest": 2}),
        ({}, {"beam_size": 1}),
        ({}, {"beam_size": 2}),
        ({}, {"beam_size": 2, "search_type": "nsc"}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 2, "prefix_alpha": 1}),
        ({}, {"beam_size": 2, "search_type": "tsd"}),
        ({}, {"beam_size": 2, "search_type": "tsd", "max-sym-exp": 3}),
        ({}, {"beam_size": 2, "search_type": "alsd"}),
        ({}, {"beam_size": 2, "search_type": "alsd", "u_max": 10}),
        (
            {},
            {
                "beam_size": 2,
                "search_type": "default",
                "rnnlm": get_wordlm(),
                "lm_weight": 1.0,
            },
        ),
        ({}, {"beam_size": 2, "search_type": "nsc", "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "rnnlm": get_wordlm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 2, "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "nsc", "nstep": 2, "rnnlm": get_wordlm()}),
        (
            {},
            {
                "beam_size": 2,
                "search_type": "alsd",
                "rnnlm": get_lm(),
                "lm_weight": 0.2,
            },
        ),
        (
            {},
            {
                "beam_size": 2,
                "search_type": "alsd",
                "rnnlm": get_wordlm(),
                "lm_weight": 0.6,
            },
        ),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_lm()}),
        ({}, {"beam_size": 2, "search_type": "tsd", "rnnlm": get_wordlm()}),
    ],
)
def test_pytorch_transducer_trainable_and_decodable(train_dic, recog_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()

    train_args = get_default_train_args(**train_dic)
    recog_args = get_default_recog_args(**recog_dic)

    model = E2E(idim, odim, train_args)

    batch = prepare_inputs(idim, odim, ilens, olens)

    # to avoid huge training time, cer/wer report
    # is only enabled at validation steps
    if train_args.report_cer or train_args.report_wer:
        model.training = False

    loss = model(*batch)
    loss.backward()

    beam_search = BeamSearchTransducer(
        decoder=model.dec,
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
        in_data = np.random.randn(20, idim)

        model.recognize(in_data, beam_search)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize("trans_type", ["warp-transducer", "warp-rnnt"])
def test_pytorch_transducer_gpu_trainable(trans_type):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(trans_type=trans_type)

    if trans_type == "warp-rnnt" and torch.version.cuda != "10.0":
        with pytest.raises(ImportError):
            model = E2E(idim, odim, train_args)

        return

    model = E2E(idim, odim, train_args)

    model.cuda()

    batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=True)

    loss = model(*batch)
    loss.backward()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "train_dic",
    [
        {"report_cer": True, "report_wer": True},
    ],
)
@pytest.mark.execution_timeout(2.8)
def test_pytorch_multi_gpu_trainable(train_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(**train_dic)

    ngpu = 2
    device_ids = list(range(ngpu))

    model = E2E(idim, odim, train_args)
    model = torch.nn.DataParallel(model, device_ids)
    model.cuda()

    batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=True)

    loss = 1.0 / ngpu * model(*batch)
    loss.backward(loss.new_ones(ngpu))


def test_calculate_plot_attention():
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args()

    model = E2E(idim, odim, train_args)

    batch = prepare_inputs(idim, odim, ilens, olens, is_cuda=False)

    assert model.calculate_all_attentions(*batch) == []


@pytest.mark.parametrize(
    "train_dic",
    [
        {"elayers": 2, "aux_task_type": "default", "aux_task_layer_list": [0]},
        {
            "etype": "vggblstm",
            "elayers": 3,
            "aux_task_type": "symm_kl_div",
            "aux_task_layer_list": [0, 1],
        },
        {
            "etype": "blstm",
            "elayers": 2,
            "aux_task_type": "both",
            "aux_task_layer_list": [0],
        },
        {"elayers": 2, "aux_ctc": True, "aux_ctc_weight": 0.5},
        {"elayers": 2, "aux_cross_entropy": True, "aux_cross_entropy_weight": 0.5},
    ],
)
def test_auxiliary_task(train_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()

    train_args = get_default_train_args(**train_dic)
    recog_args = get_default_recog_args()

    model = E2E(idim, odim, train_args)

    batch = prepare_inputs(idim, odim, ilens, olens)

    loss = model(*batch)
    loss.backward()

    beam_search = BeamSearchTransducer(
        decoder=model.dec,
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
                (idim, odim, vars(train_args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )

    with torch.no_grad():
        in_data = np.random.randn(20, idim)

        model, _ = load_trained_model(tmpdir + "/model.dummy.best", training=False)

        model.recognize(in_data, beam_search)


def test_invalid_aux_task_layer_list():
    idim, odim, ilens, olens = get_default_scope_inputs()
    train_args = get_default_train_args(aux_task_type="default")

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)

    train_args = get_default_train_args(
        aux_task_type="default", aux_task_layer_list="foo"
    )

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)

    train_args = get_default_train_args(
        aux_task_type="default", aux_task_layer_list=[0, 4]
    )

    with pytest.raises(ValueError):
        E2E(idim, odim, train_args)
