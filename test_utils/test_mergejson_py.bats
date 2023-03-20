#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    mkdir -p $tmpdir/input $tmpdir/output $tmpdir/other
    python << EOF
# coding: UTF-8
import json

with open('$tmpdir/input/feat.json', 'w') as f:
    d = {'utts': {'uttid': {'feat': 'aaa.ark:123'}}}
    json.dump(d, f)
with open('$tmpdir/input/ilen.json', 'w') as f:
    d = {'utts': {'uttid': {'ilen': '100'}}}
    json.dump(d, f)
with open('$tmpdir/input/idim.json', 'w') as f:
    d = {'utts': {'uttid': {'idim': '80'}}}
    json.dump(d, f)
with open('$tmpdir/output/text.json', 'w') as f:
    d = {'utts': {'uttid': {'text': 'あいうえお'}}}
    json.dump(d, f)
with open('$tmpdir/output/token.json', 'w') as f:
    d = {'utts': {'uttid': {'token': 'あ い う え お'}}}
    json.dump(d, f)
with open('$tmpdir/output/tokenid.json', 'w') as f:
    d = {'utts': {'uttid': {'tokenid': '0 1 2 3 4'}}}
    json.dump(d, f)
with open('$tmpdir/output/olen.json', 'w') as f:
    d = {'utts': {'uttid': {'olen': '10'}}}
    json.dump(d, f)
with open('$tmpdir/output/odim.json', 'w') as f:
    d = {'utts': {'uttid': {'odim': '26'}}}
    json.dump(d, f)
with open('$tmpdir/other/utt2spk.json', 'w') as f:
    d = {'utts': {'uttid': {'utt2spk': 'foobar'}}}
    json.dump(d, f)

EOF

    cat << EOF > $tmpdir/valid
{
    "utts": {
        "uttid": {
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
                        26
                    ],
                    "text": "あいうえお",
                    "token": "あ い う え お",
                    "tokenid": "0 1 2 3 4"
                }
            ],
            "utt2spk": "foobar"
        }
    }
}
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "mergejson.py" {
    python $utils/mergejson.py --input-jsons $tmpdir/input/*.json --output-jsons $tmpdir/output/*.json --jsons $tmpdir/other/*.json > $tmpdir/out.json
    jsondiff $tmpdir/out.json $tmpdir/valid
}
