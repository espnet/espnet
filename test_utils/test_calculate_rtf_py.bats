#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet2-test-calculate-rtf-${RANDOM}
    # Create dummy data
    mkdir -p ${tmpdir}/asr_results/logdir
    cp test_utils/asr_inference.1.log ${tmpdir}/asr_results/logdir/ 
    _sample_shift=$(python3 -c "print(1 / 16000 * 1000)") # in ms   

    cat << EOF > $tmpdir/valid
Total audio duration: 693.700 [sec]
Total decoding time: 5186.688 [sec]
RTF: 7.477
Latency: 63252.293 [ms/sentence]
EOF
}

@test "calculate_rtf" {
    _logdir="${tmpdir}/asr_results/logdir"
    utils/calculate_rtf.py \
        --log-dir ${_logdir} \
        --log-name "asr_inference" \
        --input-shift ${_sample_shift} \
        --start-times-marker "speech length" \
        --end-times-marker "best hypo" \
        > "${_logdir}"/calculate_rtf.log

    diff "$tmpdir/valid" "${_logdir}"/calculate_rtf.log
}

teardown() {
    rm -r $tmpdir
}
