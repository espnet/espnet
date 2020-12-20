#!/bin/bash

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
use_phoneme_text=false

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${LIBRITTS}" ]; then
   log "Fill the value of 'LIBRITTS' of db.sh"
   exit 1
fi
db_root=${LIBRITTS}
data_url=www.openslr.org/resources/60

train_set=tr_no_dev
dev_set=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/donwload_and_untar.sh"
    # download the original corpus
    if [ ! -e "${db_root}"/LibriTTS/.complete ]; then
        for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
            local/download_and_untar.sh "${db_root}" "${data_url}" "${part}"
        done
        touch "${db_root}/LibriTTS/.complete"
    else
        log "Already done. Skiped."
    fi

    # download the additional labels
    if [ ! -e "${db_root}"/LibriTTS/.lab_complete ]; then
        git clone https://github.com/kan-bayashi/LibriTTSCorpusLabel.git "${db_root}/LibriTTSCorpusLabel"
        cat "${db_root}"/LibriTTSCorpusLabel/lab.tar.gz-* > "${db_root}/LibriTTS/lab.tar.gz"
        cwd=$(pwd)
        cd "${db_root}/LibriTTS"
        for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
            gunzip -c lab.tar.gz | tar xvf - "lab/phone/${part}" --strip-components=2
        done
        touch .lab_complete
        rm -rf lab.tar.gz
        cd "${cwd}"
    else
        log "Already done. Skiped."
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    for name in train-clean-100 train-clean-360 dev-clean test-clean; do
        local/data_prep.sh "${db_root}/LibriTTS/${name}" "data/${name}"
        local/prep_segments.py --replace_text_with_phoneme ${use_phoneme_text} "data/${name}/wav.scp"
        utils/fix_data_dir.sh "data/${name}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: utils/combine_data.sh"
    utils/combine_data.sh "data/${train_set}" data/train-clean-100 data/train-clean-360
    utils/copy_data_dir.sh data/dev-clean "data/${dev_set}"
    utils/copy_data_dir.sh data/test-clean "data/${eval_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
