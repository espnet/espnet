#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# kosp2e_root: Set root to espnet/egs2/kosp2e/asr1 at current repo location
kosp2e_root=$(cd "$(dirname "$0")/.." && pwd)

# output_dir: Directory egs2/kosp2e/asr1/data
output_dir=$(cd "$(dirname "$0")/.." && pwd)/data

log "kosp2e_root: ${kosp2e_root}"
log "output_dir: ${output_dir}"

log "Stage 1: KOSP2E Data Preparation"
python3 local/setup_data.py \
    --kosp2e_root "${kosp2e_root}" \
    --output_dir "${output_dir}"

log "Successfully finished. [elapsed=${SECONDS}s]"
