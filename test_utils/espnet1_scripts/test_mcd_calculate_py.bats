#!/usr/bin/env bats
# Copyright 2020 Nagoya University (Wen-Chin Huang)

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/valid
number of utterances = 2
st_test 0.0
ctc_align_test 0.0
Mean MCD: 0.00
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "mcd_calculate.py" {
    python $utils/mcd_calculate.py --wavdir test_utils/ --gtwavdir test_utils/ --f0min 40 --f0max 700  > $tmpdir/out
    diff $tmpdir/out $tmpdir/valid
}
