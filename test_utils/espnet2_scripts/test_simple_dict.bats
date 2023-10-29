setup() {
    load "../bats-support/load"
    load "../bats-assert/load"

    utils=$BATS_TEST_DIRNAME/../../egs2/TEMPLATE/asr1/scripts/utils
    echo $utils
}

@test "simple_dict.sh: can import" {
    source $utils/simple_dict.sh
}

@test "simple_dict.sh: check number of arguments" {
    source $utils/simple_dict.sh
    dict_init d
    dict_put d k v
    dict_get d k
    dict_remove d k
    dict_keys d
    dict_values d

    run bash -c "source $utils/simple_dict.sh; dict_init"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d d"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v v"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v; dict_get"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v; dict_get d"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v; dict_get d k k"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v; dict_remove"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v; dict_remove d"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d k v; dict_remove d k k"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_keys"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_keys d d"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_values"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_values d d"
    assert_failure
}

@test "simple_dict.sh: check various dict_init edge cases" {
    run bash -c "source $utils/simple_dict.sh; dict_init good_argument"
    assert_success

    run bash -c "source $utils/simple_dict.sh; dict_init bad number_of_arguments"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init bad/argument"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init \"bad argument\""
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init duplicate; dict_init duplicate"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init duplicate; dict_init not_duplicate"
    assert_success
}

@test "simple_dict.sh: key existence test" {
    run bash -c "source $utils/simple_dict.sh; dict_get d b"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_remove d b"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_get d b"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_remove d b"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d a 1; dict_get d b"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d a 1; dict_remove d b"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d a 1; dict_get d a"
    assert_success
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d a 1; dict_remove d a"
    assert_success
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d a 1; dict_remove d a; dict_get d a"
    assert_failure
}

@test "simple_dict.sh: check various name error cases for key and value" {
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d key value"
    assert_success

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d \"key space\" value"
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d \"key\/slash\" value"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d key \"value space\""
    assert_failure
    run bash -c "source $utils/simple_dict.sh; dict_init d; dict_put d key \"value\/slash\""
    assert_failure
}

@test "simple_dict.sh: key read, write, and delete" {
    source $utils/simple_dict.sh
    dict_init d

    dict_put d a one
    assert_equal "$(dict_get d a)" "one"

    dict_put d b 2
    assert_equal "$(dict_get d a)" "one"
    assert_equal "$(dict_get d b)" "2"

    dict_put d a 1
    assert_equal "$(dict_get d a)" "1"
    assert_equal "$(dict_get d b)" "2"

    dict_remove d a
    assert_equal "$(dict_get d b)" "2"
}

@test "simple_dict.sh: traverse by sorted key order" {
    source $utils/simple_dict.sh
    dict_init d

    dict_put d a a_val
    assert_equal "$(dict_keys d)" "a"
    assert_equal "$(dict_values d)" "a_val"

    dict_put d b b_val
    assert_equal "$(dict_keys d)" "a b"
    assert_equal "$(dict_values d)" "a_val b_val"

    dict_put d aa aa_val
    assert_equal "$(dict_keys d)" "a aa b"
    assert_equal "$(dict_values d)" "a_val aa_val b_val"

    dict_put d b b_val_new
    assert_equal "$(dict_keys d)" "a aa b"
    assert_equal "$(dict_values d)" "a_val aa_val b_val_new"

    dict_remove d a
    assert_equal "$(dict_keys d)" "aa b"
    assert_equal "$(dict_values d)" "aa_val b_val_new"
}
