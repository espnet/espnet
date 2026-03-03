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

cd ./egs3/mini_an4/asr
gen_dummy_coverage
echo "==== [ESPnet3] ASR ===="
source ./path.sh
run_with_train_config() {
    local train_config=$1
    local runner=$2
    local infer_config=$3

    ln -sfn "${train_config}" conf/training.yaml
    ${python} "${runner}" \
        --stages create_dataset train_tokenizer collect_stats train infer measure \
        --train_config conf/training.yaml \
        --infer_config "${infer_config}" \
        --measure_config conf/measure.yaml
    rm -rf exp data
}

debug_configs=(
    training_asr_rnn_data_aug.yaml
    training_asr_rnn.yaml
    training_asr_streaming.yaml
    training_asr_transformer.yaml
    training_asr_transducer.yaml
)

for train_config in "${debug_configs[@]}"; do
    run_with_train_config "${train_config}" run.py conf/inference.yaml
done

# We need seprate inference config for transducer task
run_with_train_config \
    training_transducer_asr_conformer_rnnt.yaml \
    run.py \
    conf/inference_transducer.yaml

cd "${cwd}"
