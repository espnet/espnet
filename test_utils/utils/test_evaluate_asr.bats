#!/usr/bin/env bats

setup() {
    tmpdir=/tmp/espnet2-test-evaluate-asr-${RANDOM}
    # Create dummy data
    mkdir -p ${tmpdir}/data
    echo "dummy A" > ${tmpdir}/data/text
    echo "dummy ${tmpdir}/data/dummy.wav" > ${tmpdir}/data/wav.scp
    python << EOF
import numpy as np
import soundfile as sf
sf.write("${tmpdir}/data/dummy.wav", np.zeros(16000 * 2,), 16000, "PCM_16")
EOF
}

@test "evaluate_asr" {
    cd egs2/mini_an4/asr1
    model_tag="espnet/kamo-naoyuki-mini_an4_asr_train_raw_bpe_valid.acc.best"
    scripts/utils/evaluate_asr.sh \
        --stop-stage 3 \
        --model_tag "${model_tag}" \
        --gt_text "${tmpdir}/data/text" \
        --inference_args "--beam_size 1" \
        "${tmpdir}/data/wav.scp" "${tmpdir}/asr_results"
}

teardown() {
    rm -r $tmpdir
}
