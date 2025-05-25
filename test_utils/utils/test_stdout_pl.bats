#!/usr/bin/env bats
# -*- mode:sh -*-

setup() {
    tmpdir=$(mktemp -d test_stdout_pl.XXXXXX)
    echo $tmpdir
}

teardown() {
    rm -rf $tmpdir
}

@test "stdout_pl" {
    stdout=$(./utils/stdout.pl $tmpdir/log echo hi)
    [ "$stdout" = "hi" ]

    grep -R "# echo hi" $tmpdir/log
    grep -R "# Ended (code 0) at" $tmpdir/log
}

@test "stdout_pl_array" {
    stdout=$(./utils/stdout.pl JOB=1:3 $tmpdir/log.JOB echo hi.JOB)
    for i in 1 2 3; do
        [ "$(echo $stdout | grep hi.${i} )" != "" ]

        grep -R "# echo hi.${i}" $tmpdir/log.${i}
        grep -R "# Ended (code 0) at" $tmpdir/log.${i}
    done
}
