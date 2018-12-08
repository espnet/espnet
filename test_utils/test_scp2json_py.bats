#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    testdir=$(mktemp -d testXXX)
    cat << EOF >> $testdir/test.scp
uttid1 あ い う
uttid2 え お
uttid3 か き く け こ
EOF

    cat << EOF > $testdir/valid
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
    rm -rf $testdir
}

@test "" {
    cat $testdir/test.scp | python $utils/scp2json.py --key text  > $testdir/out.json
    jsondiff $testdir/out.json $testdir/valid
}
