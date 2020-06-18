#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    cat << EOF > $tmpdir/feats.scp
uttid1 /some/path/a.ark
uttid2 /some/path/b.ark
uttid3 /some/path/c.ark
EOF

    cat << EOF > $tmpdir/feats_shape
uttid1 10,80
uttid2 3,80
uttid3 4,80
EOF

    cat << EOF > $tmpdir/feats2.scp
uttid1 /some/path/d.ark
uttid2 /some/path/e.ark
uttid3 /some/path/f.ark
EOF

    cat << EOF > $tmpdir/feats2_shape
uttid1 100
uttid2 30
uttid3 20
EOF
    cat << EOF > $tmpdir/token
uttid1 あ い う
uttid2 え お
uttid3 か き く け こ
EOF

    cat << EOF > $tmpdir/token_id
uttid1 0 1 2
uttid2 3 4
uttid3 5 5 7
EOF

    cat << EOF > $tmpdir/text_shape
uttid1 3,31
uttid2 2,31
uttid3 3,31
EOF

    cat << EOF > $tmpdir/spk2utt
uttid1 A
uttid2 B
uttid3 C
EOF
    cat << EOF > $tmpdir/spk2lang
uttid1 a
uttid2 b
uttid3 c
EOF

    cat << EOF > $tmpdir/valid

   {
       "utts": {
           "uttid1": {
               "input": [
                   {
                       "feat": "/some/path/a.ark",
                       "name": "input1",
                       "shape": [
                           10,
                           80
                       ]
                   },
                   {
                       "feat": "/some/path/d.ark",
                       "name": "input2",
                       "shape": [
                           100
                       ]
                   }
               ],
               "output": [
                   {
                       "name": "target1",
                       "shape": [
                           3,
                           31
                       ],
                       "token": "あ い う",
                       "token_id": "0 1 2"
                   }
               ],
               "spk2lang": "a",
               "spk2utt": "A"
           },
           "uttid2": {
               "input": [
                   {
                       "feat": "/some/path/b.ark",
                       "name": "input1",
                       "shape": [
                           3,
                           80
                       ]
                   },
                   {
                       "feat": "/some/path/e.ark",
                       "name": "input2",
                       "shape": [
                           30
                       ]
                   }
               ],
               "output": [
                   {
                       "name": "target1",
                       "shape": [
                           2,
                           31
                       ],
                       "token": "え お",
                       "token_id": "3 4"
                   }
               ],
               "spk2lang": "b",
               "spk2utt": "B"
           },
           "uttid3": {
               "input": [
                   {
                       "feat": "/some/path/c.ark",
                       "name": "input1",
                       "shape": [
                           4,
                           80
                       ]
                   },
                   {
                       "feat": "/some/path/f.ark",
                       "name": "input2",
                       "shape": [
                           20
                       ]
                   }
               ],
               "output": [
                   {
                       "name": "target1",
                       "shape": [
                           3,
                           31
                       ],
                       "token": "か き く け こ",
                       "token_id": "5 5 7"
                   }
               ],
               "spk2lang": "c",
               "spk2utt": "C"
           }
       }
   }

EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "merge_scp2json.py" {
    python $utils/merge_scp2json.py \
        --input-scps feat:$tmpdir/feats.scp shape:$tmpdir/feats_shape:shape \
        --input-scps feat:$tmpdir/feats2.scp shape:$tmpdir/feats2_shape:shape \
        --output-scps token:$tmpdir/token token_id:$tmpdir/token_id shape:$tmpdir/text_shape:shape \
        --scps spk2utt:$tmpdir/spk2utt spk2lang:$tmpdir/spk2lang \
        -O $tmpdir/out.json
    [ "$(jsondiff $tmpdir/out.json $tmpdir/valid)" = "{}" ]
}
