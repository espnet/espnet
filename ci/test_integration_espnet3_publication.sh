#!/usr/bin/env bash

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)

training_config=${1:-conf/training_asr_transformer.yaml}
inference_config=${2:-conf/inference.yaml}
publication_config=${3:-conf/publication.yaml}
dataset_split=${4:-test}

gen_dummy_coverage() {
    touch empty.py
    ${python} empty.py
}

python3 -m pip install -e '.[asr]'

cd ./egs3/mini_an4/asr || exit
gen_dummy_coverage
echo "==== [ESPnet3] Publication ===="
source path.sh
rm -rf exp data

${python} run.py \
    --stages create_dataset train_tokenizer collect_stats train infer pack_model \
    --training_config "${training_config}" \
    --inference_config "${inference_config}" \
    --publication_config "${publication_config}"

pack_dir=$(find exp -mindepth 2 -maxdepth 2 -type d -name model_pack | sort | head -n 1)
if [ -z "${pack_dir}" ]; then
    echo "Packed model directory not found under egs3/mini_an4/asr/exp" >&2
    exit 1
fi

PACK_DIR="${pack_dir}" python3 "${cwd}/ci/test_integration_espnet3_publication_check.py" \
    --split "${dataset_split}" \
    --recipe-dir "$(pwd)"

rm -rf exp data
cd "${cwd}"
