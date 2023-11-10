#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet2-test-calculate-eerdcf-${RANDOM}
    mkdir -p ${tmpdir}
    cp test_utils/spk_trial_scores ${tmpdir}/spk_trial_scores

    # Create reference data for "calculate_eermindcf"
    cat << EOF > $tmpdir/expected_output
trg_mean: -0.856322466135025, trg_std: 0.08600128103334274
nontrg_mean: 0.08447770031926187, nontrg_std: 0.08447770031926187
eer: 2.0, mindcf: 0.04
EOF

    # Create reference data for "calculate_eermindcf_eer0"
    cat << EOF > $tmpdir/expected_output_eer0
trg_mean: 0.95, trg_std: 0.04999999999999999
nontrg_mean: 0.05, nontrg_std: 0.05
eer: 0.0, mindcf: 0.0
EOF

    # Create score files for "calculate_eermindcf_eer0"
    cat << EOF > $tmpdir/scores_eer0
trial1 1.0 1
trial2 0.0 0
trial3 0.9 1
trial4 0.1 0
EOF

    # Create reference data for "calculate_eermindcf_eer100"
    cat << EOF > $tmpdir/expected_output_eer100
trg_mean: 0.05, trg_std: 0.05
nontrg_mean: 0.04999999999999999, nontrg_std: 0.04999999999999999
eer: 100.0, mindcf: 1.0
EOF

    # Create score files for "calculate_eermindcf_eer100"
    cat << EOF > $tmpdir/scores_eer100
trial1 1.0 0
trial2 0.0 1
trial3 0.9 0
trial4 0.1 1
EOF
}

@test "calculate_eermindcf" {
    python egs2/TEMPLATE/asr1/pyscripts/utils/calculate_eer_mindcf.py \
    ${tmpdir}/spk_trial_scores ${tmpdir}/calculated_output

    diff "${tmpdir}"/expected_output "${tmpdir}"/calculated_output
}

@test "calculate_eermindcf_eer0" {
    python egs2/TEMPLATE/asr1/pyscripts/utils/calculate_eer_mindcf.py \
    ${tmpdir}/scores_eer0 ${tmpdir}/calculated_output

    diff "${tmpdir}"/expected_output_eer0 "${tmpdir}"/calculated_output
}

@test "calculate_eermindcf_eer100" {
    python egs2/TEMPLATE/asr1/pyscripts/utils/calculate_eer_mindcf.py \
    ${tmpdir}/scores_eer100 ${tmpdir}/calculated_output

    diff "${tmpdir}"/expected_output_eer100 "${tmpdir}"/calculated_output
}

teardown() {
    rm -r $tmpdir
}
