#!/usr/bin/env bash

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)
check_workdir=""
clone_workdir=""

training_config=${1:-conf/training_asr_transformer.yaml}
inference_config=${2:-${cwd}/test_utils/egs3/integration_test/esp2_asr/conf/inference_publication.yaml}
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

cleanup() {
    if [ -n "${clone_workdir}" ] && [ -d "${clone_workdir}" ]; then
        rm -rf "${clone_workdir}"
    fi
    if [ -n "${check_workdir}" ] && [ -d "${check_workdir}" ]; then
        rm -rf "${check_workdir}"
    fi
}

trap cleanup EXIT

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

clone_workdir=$(mktemp -d)
espnet3 clone mini_an4/esp2_asr --project "${clone_workdir}/recipe"
cd "${clone_workdir}/recipe" || exit
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
    echo "Packed model directory not found under the cloned recipe exp" >&2
    exit 1
fi

check_workdir="$(mktemp -d "${TMPDIR:-/tmp}/espnet3-publication-check.XXXXXX")"
if [ -e "${check_workdir}/data" ]; then
    echo "Temporary check directory unexpectedly contains data/: ${check_workdir}" >&2
    exit 1
fi

check_args=(
    --split "${dataset_split}"
    --reference-file "${check_workdir}/native-reference.json"
)
if [ -n "${hf_repo}" ]; then
    check_args+=(--model-tag "${hf_repo}")
fi

# Run the check from a directory outside the recipe tree to catch accidental
# relative-path dependencies such as ./data.  Use a subshell so the recipe
# directory remains the working directory for all subsequent commands.
(
    pack_dir_abs="$(pwd)/${pack_dir}"
    cd "${check_workdir}" || exit 1
    PACK_DIR="${pack_dir_abs}" python3 "${cwd}/ci/test_integration_espnet3_publication_check.py" \
        "${check_args[@]}"
)

${python} run.py \
    --stages pack_demo \
    --training_config "${training_config}" \
    --demo_config "${cwd}/test_utils/egs3/integration_test/esp2_asr/conf/demo_integration_default.yaml"

${python} run.py \
    --stages pack_demo \
    --training_config "${training_config}" \
    --demo_config "${cwd}/test_utils/egs3/integration_test/esp2_asr/conf/demo_integration_custom.yaml"

${python} -m playwright install chromium
${python} "${cwd}/ci/test_demo_ui.py" --reference-file "${check_workdir}/native-reference.json"

rm -rf exp data
cd "${cwd}" || exit 1
