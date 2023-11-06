#!/usr/bin/env bash
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

# Copy
for f in cmd.sh conf local; do
    target="${dir}"/../../TEMPLATE/tts1/"${f}"
    cp -r "${target}" "${dir}"
    targets+="${dir}/${target} "
done


# Symlinks to TEMPLATE/tts1
for f in tts.sh path.sh sid; do
    target=../../TEMPLATE/tts1/"${f}"
    ln -sf "${target}" "${dir}"
    targets+="${dir}/${target} "
done


# Symlinks to TEMPLATE/asr1
for f in db.sh scripts pyscripts utils steps; do
    target=../../TEMPLATE/asr1/"${f}"
    ln -sf "${target}" "${dir}"
    targets+="${dir}/${target} "
done

log "Created: ${targets}"
