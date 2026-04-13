#!/usr/bin/env bash

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)

gen_dummy_coverage(){
    touch empty.py
    ${python} empty.py
}

python3 -m pip install -e '.[asr]'

cd ./egs3/mini_an4/asr || exit
gen_dummy_coverage
echo "==== [ESPnet3] ASR ===="
source path.sh
run_with_training_config() {
    local training_config=$1
    local runner=$2
    local inference_config=$3

    ln -sfn "${training_config}" conf/training.yaml
    ${python} "${runner}" \
        --stages create_dataset train_tokenizer collect_stats train infer measure \
        --training_config conf/training.yaml \
        --inference_config "${inference_config}" \
        --metrics_config conf/metrics.yaml
    rm -rf exp data
}

training_configs=(
    training_asr_rnn_data_aug.yaml
    training_asr_rnn.yaml
    training_asr_streaming.yaml
    training_asr_transformer.yaml
    training_asr_transducer.yaml
)

for training_config in "${training_configs[@]}"; do
    run_with_training_config "${training_config}" run.py conf/inference.yaml
done

# We need seprate inference config for transducer task
run_with_training_config \
    training_transducer_asr_conformer_rnnt.yaml \
    run.py \
    conf/inference_transducer.yaml

cd "${cwd}"

python3 -m pip install -e '.[tts]'

cd ./egs3/mini_an4/tts
gen_dummy_coverage
echo "==== [ESPnet3] TTS ===="
run_with_tts_config() {
    local training_config=$1
    local inference_config=$2

    ln -sfn "${training_config}" conf/training.yaml
    ${python} run.py \
        --stages create_dataset collect_stats train infer \
        --training_config conf/training.yaml \
        --inference_config conf/inference.yaml
    rm -rf exp data
}

run_with_tts_config training_tacotron2.yaml inference.yaml
run_with_tts_config training_vits.yaml inference.yaml

cd "${cwd}"
