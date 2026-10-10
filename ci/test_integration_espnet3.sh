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

cd ./egs3/mini_an4/esp2_asr || exit
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

# The runs above use one worker, which is one shard: nothing is split, built in
# a worker process or merged. Run collect_stats, infer and measure again on two
# CPU workers and check the results are the one-worker runs' own.
# The streaming config applies no random augmentation, so collect_stats is
# deterministic and the two statistics can be compared exactly.
echo "==== [ESPnet3] ASR on multiple CPU workers ===="
ln -sfn training_asr_streaming.yaml conf/training.yaml
${python} run.py \
    --stages create_dataset train_tokenizer collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference_serial.yaml \
    --metrics_config conf/metrics.yaml
for training_config in \
    conf/training_asr_streaming_serial.yaml \
    conf/training_asr_streaming_parallel.yaml; do
    ${python} run.py --stages collect_stats --training_config "${training_config}"
done
${python} run.py \
    --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference_parallel.yaml \
    --metrics_config conf/metrics.yaml
python3 "${cwd}/ci/check_espnet3_parallel_workers.py" \
    --inference exp/training/inference_serial \
    --inference-parallel exp/training/inference_parallel \
    --stats exp/stats_serial \
    --stats-parallel exp/stats_parallel
rm -rf exp data

cd "${cwd}" || exit

python3 -m pip install -e '.[st]'

cd ./egs3/mini_an4/esp2_st || exit
gen_dummy_coverage
echo "==== [ESPnet3] ST ===="
source path.sh
${python} run.py \
    --stages create_dataset train_tokenizer collect_stats train infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
rm -rf exp data

cd "${cwd}" || exit
