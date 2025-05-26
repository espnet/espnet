#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/text
あ い う え お
あ あ う
か き く
EOF

    cat << EOF > $tmpdir/valid
<unk> 1
あ 2
い 3
う 4
え 5
お 6
か 7
き 8
く 9
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "text2vocabulary.py" {
    python $utils/text2vocabulary.py $tmpdir/text > $tmpdir/vocab
    diff $tmpdir/vocab $tmpdir/valid
}
