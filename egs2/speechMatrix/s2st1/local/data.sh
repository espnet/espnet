#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
src_lang=es

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${SPEECHMATRIX}" ]; then
    log "Fill the value of 'SPEECHMATRIX' of db.sh"
    exit 1
fi

set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for speechMatrix"
    for part in "train"; do
        log "Prepare speechMatrix ${part}"
        mkdir -p data/"${part}_${src_lang}"
        python local/sm_data_prep.py \
            --datadir "${SPEECHMATRIX}" \
            --dest data/"${part}_${src_lang}" \
            --src_lang ${src_lang}
            # --subset ${part} \

        ./utils/utt2spk_to_spk2utt.pl data/${part}_${src_lang}/utt2spk > data/${part}_${src_lang}/spk2utt

        ln -sf text.en data/"${part}_${src_lang}"/text
        ln -sf wav.scp.en data/"${part}_${src_lang}"/wav.scp

        utt_extra_files="wav.scp.${src_lang} wav.scp.en text.${src_lang} text.en"
        # [Ziang] no speaker information for speechMatrix, would fail the fix_data_dir.sh, creating placeholder files `utt2spk` and `spk2utt` for now.
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" data/${part}_${src_lang}
    done

    for part in "dev" "test"; do
        log "Prepare speechMatrix ${part}"
        python local/proc_eval_data.py \
            --datadir "${SPEECHMATRIX}" \
            --dest data/"${part}_${src_lang}" \
            --subset $part

        cd data/"${part}_${src_lang}"
        ln -sf text.en text
        ln -sf wav.scp.es wav.scp
        cd ../..

        ./utils/utt2spk_to_spk2utt.pl data/${part}_${src_lang}/utt2spk > data/${part}_${src_lang}/spk2utt
        utils/fix_data_dir.sh data/${part}_${src_lang}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
