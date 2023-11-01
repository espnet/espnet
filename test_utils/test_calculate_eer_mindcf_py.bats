#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet2-test-calculate-eerdcf-${RANDOM}
    mkdir -p ${tmpdir}
    cp test_utils/spk_trial_scores ${tmpdir}/spk_trial_scores

    # Create reference data
    cat << EOF > $tmpdir/expected_output
trg_mean: -0.8300532807187527, trg_std: 0.13157279774189995
nontrg_mean: 0.07914457056750071, nontrg_std: 0.07914457056750071
eer: 1.0395841663334626, mindcf: 0.05559600889536194
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
