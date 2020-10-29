#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    cd $BATS_TEST_DIRNAME/../egs/wsj/asr1
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    mkdir -p $tmpdir/data
    ark_1=$tmpdir/test_1.ark
    scp_1=$tmpdir/test_1.scp

    ark_2=$tmpdir/test_2.ark
    scp_2=$tmpdir/test_2.scp

    mkdir -p $tmpdir/data_multilingual
    ark_1_multilingual=$tmpdir/test_1_multilingual.ark
    scp_1_multilingual=$tmpdir/test_1_multilingual.scp

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

with kaldiio.WriteHelper('ark,scp:{},{}'.format('$ark_1_multilingual', '$scp_1_multilingual')) as f:
    for i in range(2):
        x = np.ones((30, 20)).astype(np.float32)
        uttid = 'uttid{}-lang{}'.format(i, i+1)
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

    cat << EOF > $tmpdir/data_multilingual/text
uttid0-lang1 <lang1> ABC ABC
uttid1-lang2 <lang2> BC BC
EOF

    cat << EOF > $tmpdir/data_multilingual/utt2spk
uttid0-lang1 spk1
uttid1-lang2 spk2
EOF

    cat << EOF > $tmpdir/dict_multilingual
<unk> 1
<space> 2
A 3
B 4
C 5
<lang1> 6
<lang2> 7
EOF

    cat << EOF > $tmpdir/nlsyms
<lang1>
<lang2>
EOF

    cat << EOF > $tmpdir/valid.json
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

    cat << EOF > $tmpdir/valid_multi_inputs.json
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

    cat << EOF > $tmpdir/valid_st.json
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
            "lang": "lang1",
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
            "lang": "lang1",
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

    cat << EOF > $tmpdir/valid_multilingual_st.json
{
    "utts": {
        "uttid0-lang1": {
            "input": [
                {
                    "feat": "${ark_1_multilingual}:7",
                    "name": "input1",
                    "shape": [
                        30,
                        20
                    ]
                }
            ],
            "lang": "lang1",
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        8,
                        9
                    ],
                    "text": "<lang1> ABC ABC",
                    "token": "<lang1> A B C <space> A B C",
                    "tokenid": "6 3 4 5 2 3 4 5"
                }
            ],
            "utt2spk": "spk1"
        },
        "uttid1-lang2": {
            "input": [
                {
                    "feat": "${ark_1_multilingual}:2429",
                    "name": "input1",
                    "shape": [
                        30,
                        20
                    ]
                }
            ],
            "lang": "lang2",
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        6,
                        9
                    ],
                    "text": "<lang2> BC BC",
                    "token": "<lang2> B C <space> B C",
                    "tokenid": "7 4 5 2 4 5"
                }
            ],
            "utt2spk": "spk2"
        }
    }
}
EOF

    cat << EOF > $tmpdir/valid_mt.json
{
    "utts": {
        "uttid0": {
            "input": [],
            "lang": "lang1",
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
            "input": [],
            "lang": "lang1",
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

    cat << EOF > $tmpdir/valid_multilingual_mt.json
{
    "utts": {
        "uttid0-lang1": {
            "input": [],
            "lang": "lang1",
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        8,
                        9
                    ],
                    "text": "<lang1> ABC ABC",
                    "token": "<lang1> A B C <space> A B C",
                    "tokenid": "6 3 4 5 2 3 4 5"
                }
            ],
            "utt2spk": "spk1"
        },
        "uttid1-lang2": {
            "input": [],
            "lang": "lang2",
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        6,
                        9
                    ],
                    "text": "<lang2> BC BC",
                    "token": "<lang2> B C <space> B C",
                    "tokenid": "7 4 5 2 4 5"
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
        $tmpdir/dict > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid.json
}

@test "data2json.sh: multi inputs" {
    $utils/data2json.sh --feat $scp_1,$scp_2 $tmpdir/data \
        $tmpdir/dict > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid_multi_inputs.json
}

@test "data2json.sh: language tag for ST and MT" {
    $utils/data2json.sh --feat $scp_1 --lang lang1 $tmpdir/data \
        $tmpdir/dict > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid_st.json
}

@test "data2json.sh: no input for MT" {
    $utils/data2json.sh --lang lang1 $tmpdir/data \
        $tmpdir/dict > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid_mt.json
}

@test "data2json.sh: multilingual ST" {
    $utils/data2json.sh --feat $scp_1_multilingual --nlsyms $tmpdir/nlsyms $tmpdir/data_multilingual \
        $tmpdir/dict_multilingual > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid_multilingual_st.json
}

@test "data2json.sh: multilingual MT" {
    $utils/data2json.sh --nlsyms $tmpdir/nlsyms $tmpdir/data_multilingual \
        $tmpdir/dict_multilingual > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid_multilingual_mt.json
}
