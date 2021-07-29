#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=100
spk=Hokuspokus
text_format=raw
threshold=35
nj=32
g2p=espeak_ng_german

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh || exit 1;
# shellcheck disable=SC1091
. ./cmd.sh || exit 1;
# shellcheck disable=SC1091
. ./db.sh || exit 1;

if [ -z "${HUI_ACG}" ]; then
   log "Fill the value of 'HUI_ACG' of db.sh"
   exit 1
fi

db_root=${HUI_ACG}
train_set=${spk,,}_tr_no_dev
dev_set=${spk,,}_dev
eval_set=${spk,,}_eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/data_download.sh"
    local/data_download.sh "${db_root}" "${spk}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}" "${spk}" "data/${spk,,}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: scripts/audio/trim_silence.sh"
    # shellcheck disable=SC2154
    scripts/audio/trim_silence.sh \
        --cmd "${train_cmd}" \
        --nj "${nj}" \
        --fs 44100 \
        --win_length 2048 \
        --shift_length 512 \
        --threshold "${threshold}" \
        "data/${spk,,}" "data/${spk,,}/log"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    utils/subset_data_dir.sh "data/${spk,,}" 500 "data/${spk,,}_deveval"
    utils/subset_data_dir.sh --first "data/${spk,,}_deveval" 250 "data/${dev_set}"
    utils/subset_data_dir.sh --last "data/${spk,,}_deveval" 250 "data/${eval_set}"
    utils/copy_data_dir.sh "data/${spk,,}" "data/${train_set}"
    utils/filter_scp.pl --exclude "data/${spk,,}_deveval/wav.scp" \
        "data/${spk,,}/wav.scp" > "data/${train_set}/wav.scp"
    utils/fix_data_dir.sh "data/${train_set}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ "${text_format}" = phn ]; then
    log "stage 3: pyscripts/utils/convert_text_to_phn.py"
    for dset in "${train_set}" "${dev_set}" "${eval_set}"; do
        utils/copy_data_dir.sh "data/${dset}" "data/${dset}_phn"
        pyscripts/utils/convert_text_to_phn.py --g2p "${g2p}" --nj "${nj}" \
            "data/${dset}/text" "data/${dset}_phn/text"
        utils/fix_data_dir.sh "data/${dset}_phn"
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
