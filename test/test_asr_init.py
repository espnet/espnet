# coding: utf-8

import argparse
import json
import os
import tempfile

import numpy as np
import pytest
import torch

import espnet.nets.pytorch_backend.lm.default as lm_pytorch

from espnet.asr.asr_utils import torch_save
from espnet.asr.pytorch_backend.asr_init import freeze_modules
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.nets_utils import pad_list


def get_rnn_args(**kwargs):
    train_defaults = dict(
        elayers=1,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=2,
        eprojs=2,
        dtype="lstm",
        dlayers=1,
        dunits=2,
        atype="location",
        aheads=1,
        awin=2,
        aconv_chans=1,
        aconv_filts=2,
        mtlalpha=1.0,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=2,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        nbest=1,
        beam_size=1,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        lm_weight=0.0,
        rnnlm=None,
        verbose=2,
        char_list=["a", "e", "i", "o", "u"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        replace_sos=False,
        tgt_lang=False,
        enc_init=None,
        enc_init_mods="enc.",
        dec_init=None,
        dec_init_mods="dec.",
        freeze_mods=None,
        model_module="espnet.nets.pytorch_backend.e2e_asr:E2E",
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def get_rnnt_args(**kwargs):
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
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_embed_decoder=0.0,
        joint_dim=2,
        joint_activation_type="tanh",
        aux_task_type=None,
        rnnt_mode="rnnt",
        trans_type="warp-transducer",
        char_list=["a", "b", "c", "d"],
        sym_space="<space>",
        sym_blank="<blank>",
        report_cer=False,
        report_wer=False,
        beam_size=1,
        nbest=1,
        verbose=0,
        outdir=None,
        rnnlm=None,
        enc_init=None,
        enc_init_mods="enc.",
        dec_init=None,
        dec_init_mods="dec.",
        freeze_mods=None,
        model_module="espnet.nets.pytorch_backend.e2e_asr_transducer:E2E",
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


def get_default_scope_inputs():
    idim = 10
    odim = 5
    ilens = [10, 6]
    olens = [4, 3]

    return idim, odim, ilens, olens


def get_lm(n_layers, n_units, char_list):
    char_list = ["<blank>"] + char_list + ["<eos>"]

    rnnlm = lm_pytorch.ClassifierWithState(
        lm_pytorch.RNNLM(len(char_list), n_layers, n_units, typ="lstm")
    )

    return rnnlm


def pytorch_prepare_inputs(idim, odim, ilens, olens, is_cuda=False):
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
    "model_type, finetune_dic",
    [
        (
            "rnn",
            {
                "enc_init": None,
                "dec_init": True,
                "dec_init_mods": "dec.,att.",
                "mtlalpha": 0.5,
                "use_lm": None,
            },
        ),
        (
            "rnnt",
            {
                "enc_init": None,
                "dec_init": True,
                "dec_init_mods": "dec.0.",
                "use_lm": True,
            },
        ),
    ],
)
def test_pytorch_trainable_and_transferable(model_type, finetune_dic):
    idim, odim, ilens, olens = get_default_scope_inputs()

    if model_type == "rnn":
        from espnet.nets.pytorch_backend.e2e_asr import E2E

        arg_function = get_rnn_args
    else:
        from espnet.nets.pytorch_backend.e2e_asr_transducer import E2E

        arg_function = get_rnnt_args

    args = arg_function()

    model = E2E(idim, odim, args)

    batch = pytorch_prepare_inputs(idim, odim, ilens, olens)

    loss = model(*batch)
    loss.backward()

    if not os.path.exists(".pytest_cache"):
        os.makedirs(".pytest_cache")

    tmppath = tempfile.mktemp()

    if finetune_dic["use_lm"] is not None:
        lm = get_lm(args.dlayers, args.dunits, args.char_list)
        tmppath += "_rnnlm"

        torch_save(tmppath, lm)
    else:
        torch_save(tmppath, model)

    if finetune_dic["enc_init"] is not None:
        finetune_dic["enc_init"] = tmppath
    if finetune_dic["dec_init"] is not None:
        finetune_dic["dec_init"] = tmppath

    finetune_args = arg_function(**finetune_dic)

    # create dummy model.json for saved model to go through
    # get_model_conf(...) called in load_trained_modules method.
    model_conf = os.path.dirname(tmppath) + "/model.json"
    with open(model_conf, "wb") as f:
        f.write(
            json.dumps(
                (idim, odim, vars(finetune_args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )

    model = load_trained_modules(idim, odim, finetune_args)

    loss = model(*batch)
    loss.backward()

    if model_type == "rnnt":
        beam_search = BeamSearchTransducer(
            decoder=model.dec,
            joint_network=model.joint_network,
            beam_size=1,
            lm=None,
            lm_weight=0.0,
            search_type="default",
            max_sym_exp=2,
            u_max=10,
            nstep=1,
            prefix_alpha=1,
            score_norm=False,
        )

        with torch.no_grad():
            in_data = np.random.randn(10, idim)
            model.recognize(in_data, beam_search)
    else:
        with torch.no_grad():
            in_data = np.random.randn(10, idim)
            model.recognize(in_data, args, args.char_list)


# todo (b-flo): add test for frozen layers
def test_pytorch_freezable():
    from espnet.nets.pytorch_backend.e2e_asr import E2E

    idim, odim, ilens, olens = get_default_scope_inputs()
    args = get_rnn_args(freeze_mods="enc.enc.0.")

    model = E2E(idim, odim, args)
    model, model_params = freeze_modules(model, args.freeze_mods)

    model.train()
