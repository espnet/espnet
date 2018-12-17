#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/input.json
{
    "utts": {
        "uttid1": {
            "input": [
                {
                    "feat": "aaa.ark:123", 
                    "text": "あ い"
                }
            ]
        },
        "uttid2": {
            "input": [
                {
                    "feat": "aaa.ark:456", 
                    "text": "か き"
                }
            ]
        },
        "uttid3": {
            "input": [
                {
                    "feat": "aaa.ark:789", 
                    "text": "さ し"
                }
            ]
        },
        "uttid4": {
            "input": [
                {
                    "feat": "aaa.ark:111111", 
                    "text": "た ち"
                }
            ]
        },
        "uttid5": {
            "input": [
                {
                    "feat": "aaa.ark:22222", 
                    "text": "な に"
                }
            ]
        }
    }
}
EOF

cat << EOF > $tmpdir/valid1
{
    "utts": {
        "uttid1": {
            "input": [
                {
                    "feat": "aaa.ark:123",
                    "text": "あ い"
                }
            ]
        },
        "uttid2": {
            "input": [
                {
                    "feat": "aaa.ark:456",
                    "text": "か き"
                }
            ]
        },
        "uttid3": {
            "input": [
                {
                    "feat": "aaa.ark:789",
                    "text": "さ し"
                }
            ]
        }
    }
}
EOF
 
cat << EOF > $tmpdir/valid2
{
    "utts": {
        "uttid4": {
            "input": [
                {
                    "feat": "aaa.ark:111111",
                    "text": "た ち"
                }
            ]
        },
        "uttid5": {
            "input": [
                {
                    "feat": "aaa.ark:22222",
                    "text": "な に"
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

@test "" {
    python $utils/splitjson.py -p 2 $tmpdir/input.json 
    jsondiff $tmpdir/split2utt/input.1.json $tmpdir/valid1
    jsondiff $tmpdir/split2utt/input.2.json $tmpdir/valid2
}

