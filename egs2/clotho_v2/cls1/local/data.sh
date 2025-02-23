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
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

task_name=$2

if [ "${task_name}" == "entailment" ]; then
    local/data_entailment.sh "$@"
elif [ "${task_name}" == "aqa" ]; then
    local/data_aqa.sh "$@"
else
    echo "Task name ${task_name} not recognized. Please choose between 'entailment' and 'aqa'."
    exit 1
fi