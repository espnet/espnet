#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    cd $BATS_TEST_DIRNAME/../egs/wsj/asr1
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    mkdir -p $tmpdir/data
    ark_1=$tmpdir/test_1.ark
    scp_1=$tmpdir/test_1.scp

    mkdir -p $tmpdir/data_multilingual
    ark_1_multilingual=$tmpdir/test_1_multilingual.ark
    scp_1_multilingual=$tmpdir/test_1_multilingual.scp

    cat << EOF > $tmpdir/data/text
uttid0 ABC ABC
uttid1 BC BC
EOF

    cat << EOF > $tmpdir/data/text_src
uttid0 CBA CBA
uttid1 CB CB
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

    cat << EOF > $tmpdir/data_multilingual/text_src
uttid0-lang1 <lang2> CBA CBA
uttid1-lang2 <lang2> CB CB
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

    cat << EOF > $tmpdir/base.json
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


    cat << EOF > $tmpdir/base_mt.json
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
                {
                    "name": "target2",
                    "shape": [
                        7,
                        7
                    ],
                    "text": "CBA CBA",
                    "token": "C B A <space> C B A",
                    "tokenid": "5 4 3 2 5 4 3"
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
                {
                    "name": "target2",
                    "shape": [
                        5,
                        7
                    ],
                    "text": "CB CB",
                    "token": "C B <space> C B",
                    "tokenid": "5 4 2 5 4"
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
                {
                    "name": "target2",
                    "shape": [
                        7,
                        7
                    ],
                    "text": "CBA CBA",
                    "token": "C B A <space> C B A",
                    "tokenid": "5 4 3 2 5 4 3"
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
                {
                    "name": "target2",
                    "shape": [
                        5,
                        7
                    ],
                    "text": "CB CB",
                    "token": "C B <space> C B",
                    "tokenid": "5 4 2 5 4"
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

@test "update_json.sh: multi outputs" {
    $utils/update_json.sh --text $tmpdir/data/text_src $tmpdir/base.json $tmpdir/data \
        $tmpdir/dict > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid.json
}

@test "update_json.sh: MT" {
    $utils/update_json.sh --text $tmpdir/data/text_src $tmpdir/base_mt.json $tmpdir/data \
        $tmpdir/dict > $tmpdir/data.json
    jsondiff $tmpdir/data.json $tmpdir/valid_mt.json
}
