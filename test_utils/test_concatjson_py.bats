#!/usr/bin/env bats

setup() {
    utils=$(dirname BATS_TEST_FILENAME)/../utils
    testdir=$(mktemp -d testXXX)
    cat << EOF > $testdir/test1.json
{"utts": {"uttid1": [{"feat": "aaa.ark:123", "text": "あ い"}]}}
EOF

    cat << EOF > $testdir/test2.json
{"utts": {"uttid2": [{"feat": "aaa.ark:456", "text": "か き"}]}}
EOF

    cat << EOF > $testdir/valid
{
    "utts": {
        "uttid1": [
            {
                "feat": "aaa.ark:123", 
                "text": "あ い"
            }
        ], 
        "uttid2": [
            {
                "feat": "aaa.ark:456", 
                "text": "か き"
            }
        ]
    }
}
EOF

}

teardown() {
    rm -rf $testdir
}

@test "" {
    python $utils/concatjson.py $testdir/*.json > $testdir/out.json
    jsondiff $testdir/out.json $testdir/valid
}

