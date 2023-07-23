setup() {
    load 'bats-support/load'
    load 'bats-assert/load'

    cd $BATS_TEST_DIRNAME/../egs2/mls/asr2
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    echo $utils
}

@test "simple_dict.sh: can import" {
    source $utils/simple_dict.sh
}

@test "simple_dict.sh: check various dict_init edge cases" {
    run bash -c "source $utils/simple_dict.sh; dict_init good_argument"
    assert_success

    run bash -c "source $utils/simple_dict.sh; dict_init bad number_of_arguments"
    assert_failure

    run bash -c "source $utils/simple_dict.sh; dict_init bad/argument"
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

@test "simple_dict.sh: key read, write, and delete" {
    source $utils/simple_dict.sh
    dict_init d

    dict_put d a one
    assert_equal 'one' $(dict_get d a)

    dict_put d b 2
    assert_equal 'one' $(dict_get d a)
    assert_equal '2' $(dict_get d b)

    dict_put d a 1
    assert_equal '1' $(dict_get d a)
    assert_equal '2' $(dict_get d b)

    dict_remove d a
    assert_equal '2' $(dict_get d b)
}
