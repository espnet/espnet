#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    cat << EOF >> $tmpdir/valid.hyp
A B C (spk1-uttid1)
A <unk> <unk> (spk2-uttid2)
EOF

    cat << EOF >> $tmpdir/valid.ref
A B C (spk1-uttid1)
A D E (spk2-uttid2)
EOF

    cat << EOF > $tmpdir/data.json
{
    "utts": {
        "uttid1": {
            "output": [
                {
                    "rec_tokenid": "3 4 5",
                    "token": "A B C"
                }
            ],
            "utt2spk": "spk1"
        },
        "uttid2": {
            "output": [
                {
                    "rec_tokenid": "3 1 1",
                    "token": "A D E"
                }
            ],
            "utt2spk": "spk2"
        }
    }
}
EOF

    cat << EOF > $tmpdir/dict
<unk> 1
<space> 2
A 3
B 4
C 5
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "json2trn.py" {
    python $utils/json2trn.py $tmpdir/data.json $tmpdir/dict --refs $tmpdir/ref --hyps $tmpdir/hyp
    diff $tmpdir/ref $tmpdir/valid.ref
    diff $tmpdir/hyp $tmpdir/valid.hyp
}
