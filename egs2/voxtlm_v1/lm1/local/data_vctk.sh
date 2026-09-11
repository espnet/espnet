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
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 1 ]; then
    log "Error: data_dir required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${VCTK}" ]; then
   log "Fill the value of 'VCTK' of db.sh"
   exit 1
fi
db_root=${VCTK}

train_set=train
dev_set=
eval_set=

vctk_dir="local/vctk"
data_dir=$1
data_dir="${data_dir}/tts"
mkdir -p ${data_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    if [ -d "${db_root}/VCTK-Corpus/wav48_silence_trimmed" ] && [ -d "${db_root}/VCTK-Corpus/txt" ]; then
        log "stage 1: ${db_root}/VCTK-Corpus already present. Skip data downloading"
    else
        # Official CSTR VCTK Corpus v0.92 (Edinburgh DataShare, DOI 10.7488/ds/2645).
        # This is the "silence-trimmed" release (wav48_silence_trimmed/), not the
        # older VCTK-Corpus (wav48/) that upstream espnet's data_download.sh fetches
        # from udialogue.org -- that older layout doesn't match what
        # data_prep_0.92.sh expects, so we pull this version specifically.
        mkdir -p "${db_root}"
        zip_path="${db_root}/VCTK-Corpus-0.92.zip"
        if [ ! -e "${zip_path}" ]; then
            log "stage 1: downloading VCTK-Corpus-0.92.zip to ${zip_path}"
            wget --continue --tries=0 --timeout=30 --waitretry=10 \
                --retry-connrefused -O "${zip_path}" \
                "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
        fi
        log "stage 1: verifying archive integrity"
        unzip -tq "${zip_path}"
        log "stage 1: extracting to ${db_root}/VCTK-Corpus"
        mkdir -p "${db_root}/VCTK-Corpus"
        unzip -q -o "${zip_path}" -d "${db_root}/VCTK-Corpus"
        rm -f "${zip_path}"
        log "stage 1: successfully downloaded and extracted VCTK-Corpus-0.92"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: local/vctk/data_prep_0.92.sh"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    ${vctk_dir}/data_prep_0.92.sh \
        --train_set "${train_set}" --dev_set "${dev_set}" --eval_set "${eval_set}" \
        --num_dev 0 --num_eval 0 \
        "${db_root}"/VCTK-Corpus "${data_dir}"
fi
