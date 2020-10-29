#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    cd $BATS_TEST_DIRNAME/../egs/wsj/asr1
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    mkdir -p $tmpdir/data
    ark_1=$tmpdir/test_1.ark

    mkdir -p $tmpdir/data_multilingual
    ark_1_multilingual=$tmpdir/test_1_multilingual.ark

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
uttid0-lang1 <lang3> CBA CBA
uttid1-lang2 <lang4> CB CB
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
<lang3> 8
<lang4> 9
EOF

    cat << EOF > $tmpdir/nlsyms
<lang1>
<lang2>
<lang3>
<lang4>
EOF

    cat << EOF > $tmpdir/base_st.json
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

    cat << EOF > $tmpdir/base_multilingual_st.json
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

    cat << EOF > $tmpdir/base_multilingual_mt.json
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
                        11
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
                        11
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
                },
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
                },
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
                },
                {
                    "name": "target2",
                    "shape": [
                        8,
                        9
                    ],
                    "text": "<lang3> CBA CBA",
                    "token": "<lang3> C B A <space> C B A",
                    "tokenid": "6 5 4 3 2 5 4 3"
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
                },
                {
                    "name": "target2",
                    "shape": [
                        6,
                        9
                    ],
                    "text": "<lang4> CB CB",
                    "token": "<lang4> C B <space> C B",
                    "tokenid": "7 5 4 2 5 4"
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
                },
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
                },
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
                        11
                    ],
                    "text": "<lang1> ABC ABC",
                    "token": "<lang1> A B C <space> A B C",
                    "tokenid": "6 3 4 5 2 3 4 5"
                },
                {
                    "name": "target2",
                    "shape": [
                        8,
                        11
                    ],
                    "text": "<lang3> CBA CBA",
                    "token": "<lang3> C B A <space> C B A",
                    "tokenid": "8 5 4 3 2 5 4 3"
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
                        11
                    ],
                    "text": "<lang2> BC BC",
                    "token": "<lang2> B C <space> B C",
                    "tokenid": "7 4 5 2 4 5"
                },
                {
                    "name": "target2",
                    "shape": [
                        6,
                        11
                    ],
                    "text": "<lang4> CB CB",
                    "token": "<lang4> C B <space> C B",
                    "tokenid": "9 5 4 2 5 4"
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
    $utils/update_json.sh --text $tmpdir/data/text_src $tmpdir/base_st.json $tmpdir/data \
        $tmpdir/dict
    jsondiff $tmpdir/base_st.json $tmpdir/valid_st.json
}

@test "update_json.sh: MT" {
    $utils/update_json.sh --text $tmpdir/data/text_src $tmpdir/base_mt.json $tmpdir/data \
        $tmpdir/dict
    jsondiff $tmpdir/base_mt.json $tmpdir/valid_mt.json
}

@test "update_json.sh: multilingual ST" {
    $utils/update_json.sh --text $tmpdir/data_multilingual/text_src --nlsyms $tmpdir/nlsyms $tmpdir/base_multilingual_st.json $tmpdir/data_multilingual \
        $tmpdir/dict_multilingual
    jsondiff $tmpdir/base_multilingual_st.json $tmpdir/valid_multilingual_st.json
}

@test "update_json.sh: multilingual MT" {
    $utils/update_json.sh --text $tmpdir/data_multilingual/text_src --nlsyms $tmpdir/nlsyms $tmpdir/base_multilingual_mt.json $tmpdir/data_multilingual \
        $tmpdir/dict_multilingual
    jsondiff $tmpdir/base_multilingual_mt.json $tmpdir/valid_multilingual_mt.json
}
