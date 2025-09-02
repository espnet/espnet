#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 <target-dir>
EOF
)


if [ $# -ne 1 ]; then
    log "${help_message}"
    log "Error: 1 positional argument is required."
    exit 2
fi


dir=$1
mkdir -p "${dir}"

if [ ! -d "${dir}"/../../TEMPLATE ]; then
    log "Error: ${dir}/../../TEMPLATE should exist. You may specify wrong directory."
    exit 1
fi

targets=""
local_files="
local/copy_data_dir.sh \
local/perturb_lid_data_dir_speed.sh \
local/prepare_ood_test.py \
local/prepare_ood_test.sh \
local/score.py
"

# Copy
for f in cmd.sh conf; do
    target="${dir}"/../../TEMPLATE/lid1/"${f}"
    cp -r "${target}" "${dir}"
    targets+="${target} "
done


# Symlinks to TEMPLATE/lid1
for f in db.sh lid.sh path.sh; do
    target=../../TEMPLATE/lid1/"${f}"
    ln -sf "${target}" "${dir}"
    targets+="${target} "
done


# Symlinks to TEMPLATE/asr1
for f in pyscripts scripts steps utils; do
    target=../../TEMPLATE/asr1/"${f}"
    ln -sf "${target}" "${dir}"
    targets+="${target} "
done


mkdir -p "${dir}"/local
for f in ${local_files}; do
    target=../../../TEMPLATE/lid1/"${f}"
    ln -sf "${target}" "${dir}/local"
    targets+="${target} "
done

cp "egs2/TEMPLATE/lid1/local/path.sh" "${dir}/local/path.sh"
targets+="${dir}/local/path.sh "

log "Created: ${targets}"
