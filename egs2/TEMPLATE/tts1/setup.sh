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
    log "Error: No positional arguments are required."
    exit 2
fi


dir=$1

log "Change directory to ${dir}"
cd "${dir}"

targets=""

# Copy
for f in cmd.sh conf; do
    target=../../TEMPLATE/tts1/"${f}"
    cp -r "${target}" .
    targets+="${dir}/${target} "
done


# Symlinks to TEMPLATE/tts1
for f in tts.sh path.sh; do
    target=../../TEMPLATE/tts1/"${f}"
    ln -sf "${target}" .
    targets+="${dir}/${target} "
done


# Symlinks to TEMPLATE/asr1
for f in scripts pyscripts; do
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
