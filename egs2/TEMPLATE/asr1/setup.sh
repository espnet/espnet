#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
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

basedir=$(echo "$dir" | cut -d'/' -f1)
num_sub=$(awk -F"/" '{print NF-1}' <<< "${dir}")

if [ "${basedir}" != "egs2" ] || [ "$num_sub" -neq 2 ]; then
    log "Error: ${dir}/../../TEMPLATE should exist. You may specify wrong directory."
    exit 1
fi


log "Change directory to ${dir}"
mkdir -p "${dir}"
cd "${dir}"

targets=""

# Copy
for f in cmd.sh conf; do
    target=../../TEMPLATE/asr1/"${f}"
    cp -r "${target}" .
    targets+="${dir}/${target} "
done


# Symlinks to TEMPLATE
for f in asr.sh path.sh db.sh scripts pyscripts; do
    target=../../TEMPLATE/asr1/"${f}"
    ln -sf "${target}" .
    targets+="${dir}/${target} "
done


# Symlinks to Kaldi
for f in steps utils; do
    target=../../../tools/kaldi/egs/wsj/s5/"${f}"
    ln -sf "${target}" .
    targets+="${dir}/${target} "
done

log "Created: ${targets}"
