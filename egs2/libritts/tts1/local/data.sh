#!/usr/bin/env bash

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
trim_all_silence=true

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
    if "${trim_all_silence}"; then
        [ ! -e data/local ] && mkdir -p data/local
        cp ${db_root}/LibriTTS/SPEAKERS.txt data/local
    fi
    for name in dev-clean test-clean train-clean-100 train-clean-360; do
        if "${trim_all_silence}"; then
            # Remove all silence and re-create wav file
            local/trim_all_silence.py "${db_root}/LibriTTS/${name}" data/local/${name}

            # Copy normalized txt files while keeping the structure
            cwd=$(pwd)
            cd "${db_root}/LibriTTS/${name}"
            find . -follow -name "*.normalized.txt" -print0 \
                | tar c --null -T - -f - | tar xf - -C "${cwd}/data/local/${name}"
            cd "${cwd}"

            # Create kaldi data directory with the trimed audio
            local/data_prep.sh "data/local/${name}" "data/${name}"
        else
            # Create kaldi data directory with the original audio
            local/data_prep.sh "${db_root}/LibriTTS/${name}" "data/${name}"
        fi
        utils/fix_data_dir.sh "data/${name}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/combine_data.sh"
    utils/combine_data.sh data/train-clean-460 data/train-clean-100 data/train-clean-360
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
