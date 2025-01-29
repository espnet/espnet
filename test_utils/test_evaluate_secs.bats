#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet2-test-evaluate-secs-${RANDOM}
    # Create dummy data
    mkdir -p ${tmpdir}/data
    echo "dummy ${tmpdir}/data/dummy.wav" > ${tmpdir}/data/wav.scp
    python << EOF
import numpy as np
import soundfile as sf
sf.write("${tmpdir}/data/dummy.wav", np.zeros(16000 * 2,), 16000, "PCM_16")
EOF
}

teardown() {
    rm -rf $tmpdir
}

@test "evaluate_secs" {
    cd egs2/mini_an4/asr1
    python pyscripts/utils/evaluate_secs.py \
        --savedir "${tmpdir}/data/pretrained_models" \
        ${tmpdir}/data/wav.scp \
        ${tmpdir}/data/wav.scp
    result="$(cut -f2 -d' ' ${tmpdir}/data/utt2secs)"
    [ "${result}" = 1.0000 ]
}
