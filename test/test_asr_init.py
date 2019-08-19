# coding: utf-8

import argparse
import importlib
import json
import os
import pytest
import tempfile


def make_arg(**kwargs):
    train_defaults = dict(
        elayers=1,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=16,
        eprojs=8,
        dtype="lstm",
        dlayers=1,
        dunits=16,
        atype="location",
        aheads=2,
        awin=5,
        aconv_chans=4,
        aconv_filts=10,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=16,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        nbest=5,
        beam_size=2,
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
        enc_init_mods='enc.',
        dec_init=None,
        dec_init_mods='dec.,att.',
        model_module='espnet.nets.pytorch_backend.e2e_asr:E2E'
    )
    train_defaults.update(kwargs)

    return argparse.Namespace(**train_defaults)


@pytest.mark.parametrize("enc_init, enc_mods, dec_init, dec_mods, mtlalpha", [
    (None, "enc.", None, "dec., att.", 0.5),
    (True, "enc.", None, "dec., att.", 0.5),
    (None, "enc.", True, "dec., att.", 0.5),
    (True, "enc.", True, "dec., att.", 0.5),
    (True, "test", None, "dec., att.", 0.5),
    (True, "test", None, "dec., att.", 0.5),
    (True, "enc.", True, "dec., att.", 0.0),
    (True, "enc.", True, "dec., att.", 1.0),
    (True, "enc.", None, "dec., att.", 0.0),
    (None, "test", True, "dec., att.", 1.0),
    (None, "enc.", True, "test", 0.5),
    (True, "test", True, "test", 0.5),
    (True, "enc.enc.0", None, "dec., att.", 0.5),
    (None, "enc.", True, "dec.embed.", 0.5),
    (True, "enc.enc.0, enc.enc.1", None, "dec., att.", 0.5),
    (None, "enc.", True, "dec.embed.,dec.decoder.1", 0.5),
    (True, "enc.enc.0, enc.enc.1", True, "dec.embed.,dec.decoder.1", 0.5)])
def test_torch_transfer_learning(enc_init, enc_mods, dec_init, dec_mods, mtlalpha):
    m = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.E2E(40, 5, args)

    if not os.path.exists(".pytest_cache"):
        os.makedirs(".pytest_cache")

    tmppath = tempfile.mktemp()
    utils.torch_save(tmppath, model)

    if enc_init:
        enc_init = tmppath
    if dec_init:
        dec_init = tmppath
    # create dummy model.json for saved model
    # to go through get_model_conf method
    model_conf = os.path.dirname(tmppath) + '/model.json'
    with open(model_conf, 'wb') as f:
        f.write(json.dumps((40, 5, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    args = make_arg(enc_init=enc_init, enc_init_mods=enc_mods,
                    dec_init=dec_init, dec_init_mods=dec_mods,
                    mtlalpha=mtlalpha)
    transfer = importlib.import_module('espnet.asr.pytorch_backend.asr_init')
    model = transfer.load_trained_modules(40, 5, args)

    os.remove(model_conf)
