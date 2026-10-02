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

cd ./egs3/mini_an4/beats || exit
echo "==== [ESPnet3] BEATs ===="
source path.sh
# Iteration 0: random-projection targets -> encoder, then codebook usage and
# the model bundle.
${python} run.py \
    --stages pretrain measure pack_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml
test -f exp/beats_iter0_tiny/model_pack/exp/beats_iter0_tiny/beats_encoder_iter0.pt
# Iteration 1: tokenizer distilled from the iteration-0 encoder -> encoder
${python} run.py \
    --stages pretrain measure \
    --training_config conf/training_iter1.yaml \
    --train_tokenizer_config conf/training_tokenizer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
test -f exp/beats_tokenizer_iter1_tiny/beats_tokenizer_iter1.pt
test -f exp/beats_iter1_tiny/beats_encoder_iter1.pt
test -f exp/beats_iter1_tiny/targets/metrics.json
rm -rf exp data downloads

cd "${cwd}" || exit
