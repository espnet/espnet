#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    cd $BATS_TEST_DIRNAME/../egs/wsj/asr1
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    mkdir -p $tmpdir/data
    ark_1=$tmpdir/test_1.ark
    scp_1=$tmpdir/test_1.scp

    ark_2=$tmpdir/test_2.ark
    scp_2=$tmpdir/test_2.scp

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import kaldiio

with kaldiio.WriteHelper('ark,scp:{},{}'.format('$ark_1', '$scp_1')) as f:
    for i in range(2):
        x = np.ones((30, 20)).astype(np.float32)
        uttid = 'uttid{}'.format(i)
        f[uttid] = x

with kaldiio.WriteHelper('ark,scp:{},{}'.format('$ark_2', '$scp_2')) as f:
    for i in range(2):
        x = np.ones((30, 20)).astype(np.float32)
        uttid = 'uttid{}'.format(i)
        f[uttid] = x
EOF

    cat << EOF > $tmpdir/data/text
uttid0 ABC ABC
uttid1 BC BC
EOF

    cat << EOF > $tmpdir/data/utt2spk
uttid0 spk1
uttid1 spk2
EOF

    cat << EOF > $tmpdir/dict
<unk> 1
<space> 2
A 3
B 4
C 5
EOF

    cat << EOF > $tmpdir/valid
{
    "utts": {
        "uttid0": {
            "input": [
                {
                    "feat": "${ark_1}:7",
                    "name": "input1",
                    "shape": [
                        30,
                        20
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        7,
                        7
                    ],
                    "text": "ABC ABC",
                    "token": "A B C <space> A B C",
                    "tokenid": "3 4 5 2 3 4 5"
                }
            ],
            "utt2spk": "spk1"
        },
        "uttid1": {
            "input": [
                {
                    "feat": "${ark_1}:2429",
                    "name": "input1",
                    "shape": [
                        30,
                        20
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        5,
                        7
                    ],
                    "text": "BC BC",
                    "token": "B C <space> B C",
                    "tokenid": "4 5 2 4 5"
                }
            ],
            "utt2spk": "spk2"
        }
    }
}
EOF

    cat << EOF > $tmpdir/valid_multi_inputs
{
    "utts": {
        "uttid0": {
            "input": [
                {
                    "feat": "${ark_1}:7",
                    "name": "input1",
                    "shape": [
                        30,
                        20
                    ]
                },
                {
                    "feat": "${ark_2}:7",
                    "name": "input2",
                    "shape": [
                        30,
                        20
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        7,
                        7
                    ],
                    "text": "ABC ABC",
                    "token": "A B C <space> A B C",
                    "tokenid": "3 4 5 2 3 4 5"
                }
            ],
            "utt2spk": "spk1"
        },
        "uttid1": {
            "input": [
                {
                    "feat": "${ark_1}:2429",
                    "name": "input1",
                    "shape": [
                        30,
                        20
                    ]
                },
                {
                    "feat": "${ark_2}:2429",
                    "name": "input2",
                    "shape": [
                        30,
                        20
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        5,
                        7
                    ],
                    "text": "BC BC",
                    "token": "B C <space> B C",
                    "tokenid": "4 5 2 4 5"
                }
            ],
            "utt2spk": "spk2"
        }
    }
}
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "data2json.sh: single input" {
    $utils/data2json.sh --feat $scp_1 $tmpdir/data \
    $tmpdir/dict > ${tmpdir}/data.json
    jsondiff ${tmpdir}/data.json $tmpdir/valid
}

@test "data2json.sh: multi inputs" {
    $utils/data2json.sh --feat $scp_1,$scp_2 $tmpdir/data \
    $tmpdir/dict > ${tmpdir}/data_multi_inputs.json
    jsondiff ${tmpdir}/data_multi_inputs.json $tmpdir/valid_multi_inputs
}
