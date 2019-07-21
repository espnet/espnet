#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    mkdir -p ${tmpdir}/{pytorch,chainer}

    # Create an ark for dummy feature
    python << EOF
import argparse
import importlib
import os
import tempfile

import numpy as np
import torch


def make_arg(**kwargs):
    defaults = dict(
        elayers=4,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=100,
        eprojs=100,
        dtype="lstm",
        dlayers=1,
        dunits=300,
        atype="location",
        aheads=4,
        awin=5,
        aconv_chans=10,
        aconv_filts=100,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=320,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        nbest=5,
        beam_size=3,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        streaming_min_blank_dur=10,
        streaming_onset_margin=2,
        streaming_offset_margin=2,
        verbose=2,
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        sortagrad=0,
        grad_noise=False,
        context_residual=False,
        use_frontend=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def make_dummy_snapshots(num_snapshots=5):
    # make torch dummy model
    m = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.E2E(40, 5, args)
    # initialize randomly
    for p in model.parameters():
        p.data.uniform_()
    for i in range(num_snapshots):
        tmppath = "${tmpdir}" + "/pytorch/snapshot.ep.%d" % (i + 1)
        torch.save({"model": model.state_dict()}, tmppath)

make_dummy_snapshots()

EOF


    cat << EOF > ${tmpdir}/log
[
    {
        "validation/main/acc": 0.1,
        "epoch": 1
    },
    {
        "validation/main/acc": 0.2,
        "epoch": 2
    },
    {
        "validation/main/acc": 0.3,
        "epoch": 3
    },
    {
        "validation/main/acc": 0.4,
        "epoch": 4
    },
    {
        "validation/main/acc": 0.5,
        "epoch": 5
    }
]
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "average_checkpoints.py: --backend pytorch " {

    python ${utils}/average_checkpoints.py --num 5 \
        --backend pytorch \
        --snapshots ${tmpdir}/pytorch/snapshot.ep.* \
        --out ${tmpdir}/pytorch/model.last5.avg.best
}

@test "average_checkpoints.py: --log --backend pytorch" {

    python ${utils}/average_checkpoints.py --num 5 \
        --log ${tmpdir}/log \
        --backend pytorch \
        --snapshots ${tmpdir}/pytorch/snapshot.ep.* \
        --out ${tmpdir}/pytorch/model.val5.avg.best
}
