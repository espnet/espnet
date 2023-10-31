#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet2-test-calculate-eerdcf-${RANDOM}
    mkdir -p ${tmpdir}
    cp test_utils/spk_trial_scores ${tmpdir}/spk_trial_scores

    # Create reference data
    cat << EOF > $tmpdir/expected_output
trg_mean: -0.809113334289397, trg_std: 0.139771755145479
nontrg_mean: 0.07949120586835534, nontrg_std: 0.07949120586835534
eer: 0.978255090648094, mindcf: 0.06860220074141932
EOF
}

@test "calculate_eermindcf" {
    python egs2/TEMPLATE/asr1/pyscripts/utils/calculate_eer_mindcf.py \
    ${tmpdir}/spk_trial_scores ${tmpdir}/calculated_output

    diff "${tmpdir}"/expected_output "${tmpdir}"/calculated_output
}

teardown() {
    rm -r $tmpdir
}
