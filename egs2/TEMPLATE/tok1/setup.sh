#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $# -ne 1 ]; then
    log "Usage: $0 <target-dir>"
    exit 2
fi

dir=$1
mkdir -p "${dir}"
if [ ! -d "${dir}/../../TEMPLATE" ]; then
    log "Error: ${dir}/../../TEMPLATE does not exist"
    exit 1
fi

# These directories are recipe-local and may be edited for each corpus.
for item in cmd.sh conf local; do
    cp -r "${dir}/../../TEMPLATE/tok1/${item}" "${dir}"
done

# These files are maintained centrally by the template.
for item in tok.sh path.sh db.sh scripts pyscripts steps utils; do
    ln -sf "../../TEMPLATE/tok1/${item}" "${dir}/${item}"
done

log "Created speech-tokenizer recipe: ${dir}"
