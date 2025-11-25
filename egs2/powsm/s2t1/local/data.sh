#!/usr/bin/env bash

# adapted from egs2/wav2gloss/asr1/local/data.sh (Kwanghee Choi)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=100
min_wav_duration=0.5
SECONDS=0

feat_dir=dump/raw

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./utils/parse_options.sh


log "data preparation started"

if [ -z "${IPAPACK_PLUS}" ]; then
    log "Fill the value of 'IPAPACK_PLUS' of db.sh"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${IPAPACK_PLUS}"

    mkdir -p ${IPAPACK_PLUS}
    local/download.sh ${IPAPACK_PLUS}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for IPAPack++ and Panphon vocab"

    # download ipa_all.csv from panphon to use as vocab list
    wget -O ipa_all.csv https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_all.csv
    if [ ! -s ipa_all.csv ] || ! head -n 1 ipa_all.csv | grep -q "syl,"; then
        log "ERROR: Failed to download or downloaded incorrect ipa_all.csv from panphon."
        rm -f ipa_all.csv
        exit 1
    fi
    cut -d',' -f1 ipa_all.csv > local/panphon_ipas
    sed -i '1d' local/panphon_ipas
    rm ipa_all.csv

    python local/data_prep.py --source_dir ${IPAPACK_PLUS} --target_dir data --min_wav_length ${min_wav_duration}

    for dir in data/*; do
        utils/fix_data_dir.sh --utt_extra_files "orthography" $dir
        utils/validate_data_dir.sh --non-print --no-feats $dir || exit 1
        # make empty files for not-yet generated extra texts
        touch $dir/text.prev $dir/text.ctc
    done
    python local/fix_doreco.py
fi

if [ ${stage} -eq 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "data prep stage 2: Additional data processing - called after stage 4"
    # generate OWSM text files for different tasks
    python local/process_ipapack.py --root_dir . --output_dir $feat_dir
    # combine text files and wav.scp for different tasks
    python local/subset.py
    for dir in $feat_dir/train $feat_dir/dev; do
        utils/fix_data_dir.sh --utt_extra_files "text.ctc text.prev" $dir
        utils/validate_data_dir.sh --non-print --no-feats $dir || exit 1
    done
    # generate list of non-linguistic symbols for BPE
    python local/generate_nlsyms.py
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
