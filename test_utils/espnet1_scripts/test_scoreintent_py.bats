#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/egs2/TEMPLATE/asr1/pyscripts/utils/
    export LC_ALL="en_US.UTF-8"
    tmpdir=$(mktemp -d testXXXXXX)/
    mkdir -p $tmpdir/valid/score_wer $tmpdir/test/score_wer
    valid_inference_folder=valid/
    test_inference_folder=test/
    echo $tmpdir
    cat <<EOF > $tmpdir/valid/score_wer/hyp.trn
decrease_heat_washroom Turn the temperature down in the bathroom	(7NqqnAOPVVSKnxyv-7NqqnAOPVVSKnxyv_01307c00-4630-11e9-bc65-55b32b211b66.wav)
decrease_heat_washroom Turn the temperature down in the washroom	(7NqqnAOPVVSKnxyv-7NqqnAOPVVSKnxyv_0157abb0-4633-11e9-bc65-55b32b211b66.wav)
EOF
    cp $tmpdir/valid/score_wer/hyp.trn $tmpdir/valid/score_wer/ref.trn
    cat <<EOF > $tmpdir/test/score_wer/hyp.trn
activate_lights_washroom Lights on in the bathroom	(4BrX8aDqK2cLZRYl-4BrX8aDqK2cLZRYl_00143870-4531-11e9-b1e4-e5985dca719e.wav)
increase_volume_none Increase the volume	(4BrX8aDqK2cLZRYl-4BrX8aDqK2cLZRYl_00224990-452e-11e9-b1e4-e5985dca719e.wav)
EOF
    cat <<EOF > $tmpdir/test/score_wer/ref.trn
activate_lights_none Lights on	(4BrX8aDqK2cLZRYl-4BrX8aDqK2cLZRYl_00143870-4531-11e9-b1e4-e5985dca719e.wav)
increase_volume_none Increase the volume	(4BrX8aDqK2cLZRYl-4BrX8aDqK2cLZRYl_00224990-452e-11e9-b1e4-e5985dca719e.wav)
EOF
    cat << EOF > $tmpdir/result.txt
Valid Intent Classification Result
1.0
Test Intent Classification Result
0.5
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "score_intent.py" {
    python $utils/score_intent.py --exp_root ${tmpdir} --valid_folder ${valid_inference_folder} --test_folder ${test_inference_folder} > $tmpdir/output.txt
    diff $tmpdir/result.txt $tmpdir/output.txt
}
