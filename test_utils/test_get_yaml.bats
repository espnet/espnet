#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)
    cat << EOF > $tmpdir/test.yaml
process:
  # these three processes are a.k.a. SpecAugument
  - type: "time_warp"
    max_time_warp: 5
    inplace: true
    mode: "PIL"
  - type: "freq_mask"
    F: 30
    n_mask: 2
    inplace: true
    replace_with_zero: false
  - type: "time_mask"
    T: 40
    n_mask: 2
    inplace: true
    replace_with_zero: false
EOF
}

teardown() {
    rm -rf $tmpdir
}


@test "get_yaml.py" {
    result1=$(python $utils/get_yaml.py $tmpdir/test.yaml process.0.type)
    [ "$result1" = "time_warp" ]

    result2=$(python $utils/get_yaml.py $tmpdir/test.yaml notexistedkey)
    [ -z "$result2" ]
}
