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

write_demo_test_model_pack_default() {
    local pack_root="exp/demo_test_model_pack_default"
    mkdir -p "${pack_root}/conf"
    cat > "${pack_root}/meta.yaml" <<'EOF'
yaml_files:
  inference_config: conf/inference.yaml
EOF
    cat > "${pack_root}/conf/inference.yaml" <<'EOF'
model:
  _target_: demo_test_inference.Inference
EOF
    # A bundle that is its own system: its model is an Inference shipped as
    # bundled code, loaded with trust_user_code. The demo shows text for speech.
    cat > "${pack_root}/demo_test_inference.py" <<'EOF'
from __future__ import annotations

from espnet3.api.inference import Field, InferenceAPI


class Inference(InferenceAPI):
    inputs = (Field("speech", "audio"),)
    outputs = (Field("text", "text"),)

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls(device=device)

    sample_rate = 16000

    def run(self, speech):
        return {"text": f"speech={int(speech is not None)}"}
EOF
}

write_demo_test_model_pack_custom() {
    local pack_root="exp/demo_test_model_pack"
    mkdir -p "${pack_root}/conf"
    cat > "${pack_root}/meta.yaml" <<'EOF'
yaml_files:
  inference_config: conf/inference.yaml
EOF
    cat > "${pack_root}/conf/inference.yaml" <<'EOF'
model:
  _target_: demo_test_inference.Inference
EOF
    # Same, plus an image input: a kind the bundle registers itself, which is
    # how a recipe adds a modality the API does not ship.
    cat > "${pack_root}/demo_test_inference.py" <<'EOF'
from __future__ import annotations

import numpy as np

from espnet3.api.inference import Field, InferenceAPI, Kind, register_kind


class ImageKind(Kind):
    def check(self, value, field, model, *, output):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{field.name} must be an image array")
        return value


register_kind("image", ImageKind(), replace=True)


class Inference(InferenceAPI):
    inputs = (Field("speech", "audio"), Field("image", "image"))
    outputs = (Field("text", "text"),)

    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls(device=device)

    sample_rate = 16000

    def run(self, speech, image):
        return {
            "text": (
                f"speech={int(speech is not None)} "
                f"image={int(image is not None)}"
            )
        }
EOF
}

cleanup() {
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

check_workdir="$(mktemp -d "${TMPDIR:-/tmp}/espnet3-publication-check.XXXXXX")"
if [ -e "${check_workdir}/data" ]; then
    echo "Temporary check directory unexpectedly contains data/: ${check_workdir}" >&2
    exit 1
fi

check_args=(
    --split "${dataset_split}"
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

write_demo_test_model_pack_default

python - "$(pwd)/exp/demo_ui_default" "$(pwd)/exp/demo_test_model_pack_default" <<'PY'
from pathlib import Path
import os
import sys
from omegaconf import OmegaConf

demo_dir = Path(sys.argv[1]).resolve()
model_dir = Path(sys.argv[2]).resolve()
config_path = demo_dir / "demo.yaml"
cfg = OmegaConf.load(config_path)
cfg.model.dir_or_tag = os.path.relpath(model_dir, start=demo_dir)
cfg.model.trust_user_code = True
config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
PY

write_demo_test_model_pack_custom

${python} run.py \
    --stages pack_demo \
    --training_config "${training_config}" \
    --demo_config "${cwd}/test_utils/egs3/integration_test/esp2_asr/conf/demo_integration_custom.yaml"

python - "$(pwd)/exp/demo_ui_custom" "$(pwd)/exp/demo_test_model_pack" <<'PY'
from pathlib import Path
import os
import sys
from omegaconf import OmegaConf

demo_dir = Path(sys.argv[1]).resolve()
model_dir = Path(sys.argv[2]).resolve()
config_path = demo_dir / "demo.yaml"
cfg = OmegaConf.load(config_path)
cfg.model.dir_or_tag = os.path.relpath(model_dir, start=demo_dir)
cfg.model.trust_user_code = True
config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
PY

${python} -m playwright install chromium
${python} "${cwd}/ci/test_demo_ui.py"

rm -rf exp data
cd "${cwd}" || exit 1
