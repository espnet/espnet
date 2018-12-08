#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXX)
    python << EOF
# coding: UTF-8
import json

with open('$tmpdir/feat.json', 'w') as f:
    d = {'utts': {'uttid': {'feat': 'aaa.ark:123'}}}
    json.dump(d, f)
with open('$tmpdir/ilen.json', 'w') as f:
    d = {'utts': {'uttid': {'ilen': '100'}}}
    json.dump(d, f)
with open('$tmpdir/idim.json', 'w') as f:
    d = {'utts': {'uttid': {'idim': '80'}}}
    json.dump(d, f)
with open('$tmpdir/text.json', 'w') as f:
    d = {'utts': {'uttid': {'text': 'あいうえお'}}}
    json.dump(d, f)
with open('$tmpdir/token.json', 'w') as f:
    d = {'utts': {'uttid': {'token': 'あ い う え お'}}}
    json.dump(d, f)
with open('$tmpdir/tokenid.json', 'w') as f:
    d = {'utts': {'uttid': {'tokenid': '0 1 2 3 4'}}}
    json.dump(d, f)
with open('$tmpdir/olen.json', 'w') as f:
    d = {'utts': {'uttid': {'olen': '10'}}}
    json.dump(d, f)
with open('$tmpdir/odim.json', 'w') as f:
    d = {'utts': {'uttid': {'odim': '26'}}}
    json.dump(d, f)
with open('$tmpdir/utt2spk.json', 'w') as f:
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

@test "" {
    python $utils/mergejson.py $tmpdir/*.json > $tmpdir/out.json
    jsondiff $tmpdir/out.json $tmpdir/valid
}

