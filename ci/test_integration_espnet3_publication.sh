#!/usr/bin/env bash

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)
check_workdir=""

training_config=${1:-conf/training_asr_transformer.yaml}
inference_config=${2:-conf/inference.yaml}
publication_config=${3:-conf/publication.yaml}
dataset_split=${4:-test}
upload_enabled=${ESPNET3_PUBLICATION_TEST_UPLOAD:-false}
publication_config_path="${publication_config}"
hf_repo=""
stages=(create_dataset train_tokenizer collect_stats train infer measure pack_model)

gen_dummy_coverage() {
    touch empty.py
    ${python} empty.py
}

resolve_hf_repo() {
    local publication_config_path=$1
    python - "${publication_config_path}" <<'PY'
from pathlib import Path
import sys

from espnet3.utils.config_utils import load_and_merge_config

config = load_and_merge_config(
    Path(sys.argv[1]),
    config_name="publication.yaml",
    default_package="egs3.TEMPLATE.esp2_asr",
    resolve=True,
)
print(getattr(getattr(config, "upload_model", None), "hf_repo", ""))
PY
}

python3 -m pip install -e '.[asr]'

cd ./egs3/mini_an4/esp2_asr || exit
gen_dummy_coverage
echo "==== [ESPnet3] Publication ===="
source path.sh
rm -rf exp data

if [ "${upload_enabled}" = "true" ]; then
    stages+=(upload_model)
    hf_repo=$(resolve_hf_repo "${publication_config_path}")
    if [ -z "${hf_repo}" ]; then
        echo "upload_model.hf_repo is empty in ${publication_config_path}" >&2
        exit 1
    fi
    echo "Using Hugging Face repo from publication config: ${hf_repo}"
fi

${python} run.py \
    --stages "${stages[@]}" \
    --training_config "${training_config}" \
    --inference_config "${inference_config}" \
    --metrics_config conf/metrics.yaml \
    --publication_config "${publication_config_path}"

pack_dir=$(find exp -mindepth 2 -maxdepth 2 -type d -name model_pack | sort | head -n 1)
if [ -z "${pack_dir}" ]; then
    echo "Packed model directory not found under egs3/mini_an4/esp2_asr/exp" >&2
    exit 1
fi

# The inference stage above executes the configured backend and output formatter.
# Publication is an artifact archive; validate its required files independently.
test -s "${pack_dir}/meta.yaml"
test -s "${pack_dir}/conf/inference.yaml"
test -s "${pack_dir}/README.md"
