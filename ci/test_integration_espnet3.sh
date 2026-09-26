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

cd "${cwd}" || exit

cd ./egs3/mini_an4/ssl || exit
echo "==== [ESPnet3] SSL (BEATs) ===="
source path.sh
# Iteration 0: random-projection targets -> encoder
${python} run.py \
    --stages create_dataset infer collect_stats train \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml
# Iteration 1: tokenizer distilled from the iteration-0 encoder -> encoder
${python} run.py \
    --stages train_tokenizer infer train \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml
test -f exp/beats_iter1_tiny/beats_encoder_iter1.pt
rm -rf exp data downloads

cd "${cwd}" || exit
