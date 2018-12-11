#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF >> $tmpdir/test.scp
uttid1 あ い う
uttid2 え お
uttid3 か き く け こ
EOF

    cat << EOF > $tmpdir/valid
{
    "utts": {
        "uttid1": {
            "text": "あ い う"
        }, 
        "uttid2": {
            "text": "え お"
        }, 
        "uttid3": {
            "text": "か き く け こ"
        }
    }
}
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "" {
    cat $tmpdir/test.scp | python $utils/scp2json.py --key text  > $tmpdir/out.json
    jsondiff $tmpdir/out.json $tmpdir/valid
}
