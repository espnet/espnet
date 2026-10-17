#! /usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=1
nj=32
lang="EN"

VALID_LANGS=("DE" "EN" "FR" "JA" "KO" "ZH")

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

found=false
for _l in "${VALID_LANGS[@]}"; do
    if [[ "${_l}" == "${lang}" ]]; then
        found=true
        break
    fi
done
if ! ${found}; then
    log "Error: Invalid language code '${lang}'. Valid options are: ${VALID_LANGS[*]}"
    exit 1
fi

if [ -z "${EMILIA}" ]; then
   log "Fill the value of 'EMILIA' of db.sh"
   exit 1
fi
db_root=${EMILIA}

train_set="tr_no_dev"
dev_set="dev"
eval_set="eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    # download the emilia and vctk corpus
    local/data_download.sh "${db_root}" "${lang}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    # format the emilia and vctk corpus to kaldi style
    local/data_prep.sh \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}" \
        --nj "${nj}" \
        --lang "${lang}" \
        "${db_root}"/emilia \
        "${db_root}"/VCTK-Corpus
fi
