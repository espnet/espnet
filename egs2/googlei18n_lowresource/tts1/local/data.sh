#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=3
threshold=35
sex=both
lang=es_ar
openslr_id=61
nj=40

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${GOOGLEI18N}" ]; then
   log "Fill the value of 'GOOGLEI18N' of db.sh"
   exit 1
fi
mkdir -p ${GOOGLEI18N}
db_root=${GOOGLEI18N}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage -1: download data from openslr"
    if [[ "${sex}" == female  ]]; then
        local/download_and_unzip.sh "${db_root}" "https://www.openslr.org/resources/${openslr_id}/${lang}_female.zip" ${lang}_female.zip
        wget -O local/line_index_female.tsv "https://www.openslr.org/resources/${openslr_id}/line_index_female.tsv"
        mv local/line_index_female.tsv local/index.tsv
    elif [[ "${sex}" == male  ]]; then
        local/download_and_unzip.sh "${db_root}" "https://www.openslr.org/resources/${openslr_id}/${lang}_male.zip" ${lang}_male.zip
        wget -O local/line_index_male.tsv "https://www.openslr.org/resources/${openslr_id}/line_index_male.tsv"
        mv local/line_index_male.tsv local/index.tsv
    else
        # local/download_and_unzip.sh "${db_root}" "https://www.openslr.org/resources/${openslr_id}/${lang}_male.zip" ${lang}_male.zip
        # local/download_and_unzip.sh "${db_root}" "https://www.openslr.org/resources/${openslr_id}/${lang}_female.zip" ${lang}_female.zip
        wget -O local/line_index_female.tsv "https://www.openslr.org/resources/${openslr_id}/line_index_female.tsv"
        wget -O local/line_index_male.tsv "https://www.openslr.org/resources/${openslr_id}/line_index_male.tsv"
        cat local/line_index_male.tsv local/line_index_female.tsv > local/index.tsv
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: prepare crowdsourced data"
    mkdir -p data
    mkdir -p data/${lang}
    log "generate utt2spk"
    awk -F '[_\t]' '{print $1 "_" $2 "_" $3 " " $1 "_" $2}' local/index.tsv > data/${lang}/utt2spk
    log "generate text"
    cp local/index.tsv data/${lang}/text
    log "generate wav.scp"
    awk -F "\t" -v db=${db_root} '{print $1 " " db}' local/index.tsv > data/${lang}/wav.scp
    log "sorting"
    sort data/${lang}/utt2spk -o data/${lang}/utt2spk
    sort data/${lang}/wav.scp -o data/${lang}/wav.scp
    sort data/${lang}/text -o data/${lang}/text
    utils/utt2spk_to_spk2utt.pl data/${lang}/utt2spk > data/${lang}/spk2utt
    utils/validate_data_dir.sh --no-feats data/${lang}

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: scripts/audio/trim_silence.sh"
        # shellcheck disable=SC2154
        scripts/audio/trim_silence.sh \
             --cmd "${train_cmd}" \
             --nj "${nj}" \
             --fs 44100 \
             --win_length 2048 \
             --shift_length 512 \
             --threshold "${threshold}" \
             data/${lang} data/${lang}/log

        utils/fix_data_dir.sh data/${lang}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: split for development set"
    utils/subset_data_dir.sh data/${lang} 500 data/dev-test-${lang}
    utils/subset_data_dir.sh data/dev-test-${lang} 250 data/dev_${lang}
    utils/copy_data_dir.sh data/dev-test-${lang} data/test_${lang}
    utils/filter_scp.pl --exclude data/dev_${lang}/wav.scp
        data/dev-test-${lang}/wav.scp > data/test_${lang}/wav.scp
    utils/fix_data_dir.sh data/test_${lang}

    utils/copy_data_dir.sh data/${lang} data/train_${lang}
    utils/filter_scp.pl --exclude data/dev-test-${lang}/wav.scp \
        data/${lang}/wav.scp > data/train_${lang}/wav.scp
    utils/fix_data_dir.sh data/train_${lang}/wav.scp
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
