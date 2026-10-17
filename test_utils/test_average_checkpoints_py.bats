#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d "${BATS_TEST_DIRNAME}/testXXXXXX")
    export tmpdir
    mkdir -p ${tmpdir}/pytorch

    # Create dummy snapshots using ESPnet2 ASR model
    python << EOF
import os
from argparse import Namespace

import torch
from espnet2.tasks.asr import ASRTask


def make_args(**kwargs):
    defaults = dict(
        encoder="rnn",
        decoder="rnn",
        input_size=40,
        encoder_conf=dict(num_layers=1, hidden_size=10),
        specaug=None,
        normalize="utterance_mvn",
        normalize_conf={},
        decoder_conf=dict(hidden_size=10, num_layers=1),
        token_list=[u"あ", u"い", u"う", u"え", u"お"],
        odim=5,
        mtlalpha=0.0,
        ctc_weight=0.2,
        ctc_conf={},
        init=None,
        ignore_id=-1,
        model_conf=dict(ctc_weight=0.2, ignore_id=-1),
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def make_dummy_snapshots(num_snapshots=5):
    tmpdir = os.environ["tmpdir"]
    args = make_args()
    model = ASRTask.build_model(args)
    for p in model.parameters():
        p.data.uniform_()
    for i in range(num_snapshots):
        tmppath = f"{tmpdir}/pytorch/snapshot.ep.{i + 1}"
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
