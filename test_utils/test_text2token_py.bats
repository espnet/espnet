#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/text
あ い う え お
あ あ う
か き く
EOF

    cat << EOF > $tmpdir/valid
 あ <space> い <space> う <space> え <space> お
 あ <space> あ <space> う
 か <space> き <space> く
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "text2token.py" {
    python $utils/text2token.py $tmpdir/text > $tmpdir/vocab
    diff $tmpdir/vocab $tmpdir/valid
}

