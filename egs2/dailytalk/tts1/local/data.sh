#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=-1
stop_stage=0
dev_dialogues=200
eval_dialogues=200

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./db.sh

if [ -z "${DAILYTALK:-}" ]; then
    log "Error: Set DAILYTALK in db.sh to the DailyTalk corpus directory."
    exit 1
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "Stage -1: Download DailyTalk corpus"

    bash local/data_download.sh "$(dirname "${DAILYTALK}")"
    if [ ! -d "${DAILYTALK}/data" ] && [ -d "$(dirname "${DAILYTALK}")/DailyTalk/data" ]; then
        DAILYTALK="$(dirname "${DAILYTALK}")/DailyTalk"
        log "Using extracted corpus directory: ${DAILYTALK}"
    fi
fi

if [ ! -d "${DAILYTALK}/data" ]; then
    log "Error: Set DAILYTALK in db.sh to a directory containing data/."
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Prepare dialogue-level train/dev/eval splits"

    split_dir=data/local/dailytalk
    mkdir -p "${split_dir}"
    find "${DAILYTALK}/data" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | sort -n > "${split_dir}/all_dialogues"

    num_dialogues=$(wc -l < "${split_dir}/all_dialogues")
    num_train=$((num_dialogues - dev_dialogues - eval_dialogues))
    if [ "${num_train}" -le 0 ]; then
        log "Error: ${num_dialogues} dialogues are insufficient for the requested splits."
        exit 1
    fi

    head -n "${num_train}" "${split_dir}/all_dialogues" > "${split_dir}/tr_no_dev"
    tail -n "$((dev_dialogues + eval_dialogues))" "${split_dir}/all_dialogues" \
        | head -n "${dev_dialogues}" > "${split_dir}/dev"
    tail -n "${eval_dialogues}" "${split_dir}/all_dialogues" > "${split_dir}/eval1"

    for set_name in tr_no_dev dev eval1; do
        out_dir="data/${set_name}"
        python3 local/data_prep.py "${DAILYTALK}" "${split_dir}/${set_name}" "${out_dir}"
        utils/utt2spk_to_spk2utt.pl "${out_dir}/utt2spk" > "${out_dir}/spk2utt"
        utils/validate_data_dir.sh --no-feats "${out_dir}"
        log "${set_name}: $(wc -l < "${out_dir}/wav.scp") utterances"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
