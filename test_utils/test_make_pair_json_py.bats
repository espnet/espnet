#!/usr/bin/env bats
# Copyright 2020 Nagoya University (Wen-Chin Huang)

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/test1.json
{
    "utts": {
        "srcspk_uttid1": {
            "input": [
                {
                    "feat": "aaa.ark:123",
                    "name": "input1",
                    "shape": [
                        100,
                        80
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        10,
                        100
                    ],
                    "text": "these quick.",
                    "token": "t h e s e <space> q u i c k  .",
                    "tokenid": "1 1 1 1 1 1 1 1 1 1"
                }
            ],
            "utt2spk": "srcspk"
        }
    }
}
EOF

    cat << EOF > $tmpdir/test2.json
{
    "utts": {
        "trgspk_uttid1": {
            "input": [
                {
                    "feat": "bbb.ark:123",
                    "name": "input1",
                    "shape": [
                        200,
                        80
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        10,
                        100
                    ],
                    "text": "these quick.",
                    "token": "t h e s e <space> q u i c k  .",
                    "tokenid": "1 1 1 1 1 1 1 1 1 1"
                }
            ],
            "utt2spk": "trgspk"
        }
    }
}
EOF

    cat << EOF > $tmpdir/valid
{
    "utts": {
        "uttid1": {
            "input": [
                {
                    "feat": "aaa.ark:123",
                    "name": "input1",
                    "shape": [
                        100,
                        80
                    ]
                }
            ],
            "output": [
                {
                    "feat": "bbb.ark:123",
                    "name": "input1",
                    "shape": [
                        200,
                        80
                    ]
                }
            ]
        }
    }
}
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "make_pair_json.py" {
    python $utils/make_pair_json.py --src-json $tmpdir/test1.json --trg-json $tmpdir/test2.json > $tmpdir/out.json
    jsondiff $tmpdir/out.json $tmpdir/valid
}
