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
${python} run.py \
    --stages create_dataset train_tokenizer collect_stats train infer measure \
    --train_config conf/train.yaml \
    --infer_config conf/infer.yaml \
    --measure_config conf/measure.yaml

rm -rf exp data
cd "${cwd}"
