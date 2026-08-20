#!/usr/bin/env bash

set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
cwd=$(pwd)
clone_workdir=""

gen_dummy_coverage(){
    touch empty.py
    ${python} empty.py
}

write_cloned_path_sh() {
    cat > path.sh <<EOF
#!/usr/bin/env bash

source "${cwd}/tools/activate_python.sh"
source "${cwd}/tools/extra_path.sh"
EOF
}

cleanup() {
    if [ -n "${clone_workdir}" ] && [ -d "${clone_workdir}" ]; then
        rm -rf "${clone_workdir}"
    fi
}

trap cleanup EXIT

python3 -m pip install -e '.[asr]'

clone_workdir=$(mktemp -d)
espnet3 clone mini_an4/asr --project "${clone_workdir}/recipe"
cd "${clone_workdir}/recipe" || exit
write_cloned_path_sh
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

# We need separate inference config for transducer task
run_with_training_config \
    training_transducer_asr_conformer_rnnt.yaml \
    run.py \
    conf/inference_transducer.yaml

cd "${cwd}" || exit 1
