#!/usr/bin/env bash

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

cwd=$(pwd)
export PYTHONPATH=${cwd}${PYTHONPATH:-}

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
}

debug_configs=train_asr_transformer_debug.yaml

echo "==== [ESPnet3] ASR Demo pack ===="
python -m pip install -e '.[asr]'
cd ./egs3/mini_an4/asr
run_with_train_config "${debug_configs}" run.py conf/infer.yaml
${python} run.py --stages pack_demo --demo_config conf/demo_ci.yaml


echo "==== [ESPnet3] Demo UI test ===="
${python} -m pytest ../../../ci/test_demo_ui.py

cd "${cwd}"
