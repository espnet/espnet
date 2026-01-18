#!/usr/bin/env bash

# set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)

cd ./egs3/mini_an4/asr
source ./path.sh

run_with_train_config() {
    local train_config=$1
    local runner=$2
    local infer_config=$3

    ln -sfn "${train_config}" conf/train.yaml
    ${python} "${runner}" \
        --stages create_dataset train_tokenizer collect_stats train infer measure \
        --train_config conf/train.yaml \
        --infer_config "${infer_config}" \
        --measure_config conf/measure.yaml
    # rm -rf exp data
}

debug_configs=train_asr_transformer_debug.yaml

echo "==== [ESPnet3] ASR Demo pack + UI ===="
run_with_train_config "${debug_configs}" run.py conf/infer.yaml
${python} run.py --stages pack_demo --demo_config conf/demo.yaml
${python} -m pytest ../../../ci/test_mini_an4_demo_ui.py

cd "${cwd}"
