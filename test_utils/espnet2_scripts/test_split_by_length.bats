#!/usr/bin/env bats

setup() {
    export LC_ALL=C

    basedir=$(cd $BATS_TEST_DIRNAME/../..; pwd)/egs2/TEMPLATE/asr1
    cd ${basedir}
    tmpdir=$(mktemp -d testXXXXXX)

    # Two long keys first, two short keys last: equal line counts put both
    # long keys into part 1, equal lengths put one long key into each part.
    printf "a a.wav\nb b.wav\nc c.wav\nd d.wav\n" > ${tmpdir}/wav.scp
    printf "a 100\nb 100\nc 1\nd 1\n" > ${tmpdir}/utt2num_samples
}

teardown() {
    cd ${basedir}
    rm -rf $tmpdir
}

@test "split_by_length: splits by total length when lengths cover the keys" {
    scripts/utils/split_by_length.sh ${tmpdir}/wav.scp ${tmpdir}/utt2num_samples \
        ${tmpdir}/out.1 ${tmpdir}/out.2
    [ "$(cut -d' ' -f1 ${tmpdir}/out.1 | tr '\n' ' ')" = "a c " ]
    [ "$(cut -d' ' -f1 ${tmpdir}/out.2 | tr '\n' ' ')" = "b d " ]
    [ "$(cat ${tmpdir}/out.1 ${tmpdir}/out.2 | sort)" = "$(cat ${tmpdir}/wav.scp)" ]
}

@test "split_by_length: --split_by_length false splits by line count" {
    scripts/utils/split_by_length.sh --split_by_length false \
        ${tmpdir}/wav.scp ${tmpdir}/utt2num_samples ${tmpdir}/out.1 ${tmpdir}/out.2
    utils/split_scp.pl ${tmpdir}/wav.scp ${tmpdir}/ref.1 ${tmpdir}/ref.2
    diff ${tmpdir}/out.1 ${tmpdir}/ref.1
    diff ${tmpdir}/out.2 ${tmpdir}/ref.2
}

@test "split_by_length: falls back to line count without a lengths file" {
    run scripts/utils/split_by_length.sh ${tmpdir}/wav.scp ${tmpdir}/missing \
        ${tmpdir}/out.1 ${tmpdir}/out.2
    [ "$status" -eq 0 ]
    [[ "$output" == *"splitting ${tmpdir}/wav.scp by line count"* ]]
    utils/split_scp.pl ${tmpdir}/wav.scp ${tmpdir}/ref.1 ${tmpdir}/ref.2
    diff ${tmpdir}/out.1 ${tmpdir}/ref.1
    diff ${tmpdir}/out.2 ${tmpdir}/ref.2
}

@test "split_by_length: falls back to line count when a key has no length" {
    printf "a 100\nb 100\nc 1\n" > ${tmpdir}/utt2num_samples
    run scripts/utils/split_by_length.sh ${tmpdir}/wav.scp ${tmpdir}/utt2num_samples \
        ${tmpdir}/out.1 ${tmpdir}/out.2
    [ "$status" -eq 0 ]
    [[ "$output" == *"1 keys of ${tmpdir}/wav.scp have no length"* ]]
    utils/split_scp.pl ${tmpdir}/wav.scp ${tmpdir}/ref.1 ${tmpdir}/ref.2
    diff ${tmpdir}/out.1 ${tmpdir}/ref.1
    diff ${tmpdir}/out.2 ${tmpdir}/ref.2
}

@test "split_by_length: rejects a value other than true or false" {
    run scripts/utils/split_by_length.sh --split_by_length yes \
        ${tmpdir}/wav.scp ${tmpdir}/utt2num_samples ${tmpdir}/out.1 ${tmpdir}/out.2
    [ "$status" -ne 0 ]
}
