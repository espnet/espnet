#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${SWBD}" ]; then
    log "Fill the value of 'SWBD' of db.sh"
    exit 1
fi
if [ ! -e "${RT03}" ]; then
    log "Fill the value of 'RT03' of db.sh"
    exit 1
fi
if [ -z "${EVAL2000}" ]; then
    log "Fill the value of 'EVAL2000' of db.sh"
    exit 1
fi

train_set=train_nodup
train_dev=train_dev
srctexts=train_nodup/text_lm

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    local/swbd1_data_download.sh ${SWBD}
    local/swbd1_prepare_dict.sh
    chmod 755 data/local/dict_nosp/lexicon0.txt
    local/swbd1_data_prep.sh ${SWBD}
    local/eval2000_data_prep.sh ${EVAL2000}
    local/rt03_data_prep.sh ${RT03}
    # use additional fisher transcriptions for LM training
    if [ -n "${FISHER}" ]; then
        local/fisher_data_prep.sh ${FISHER}
        utils/fix_data_dir.sh data/train_fisher
    fi
    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train eval2000 rt03; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    done
    # normalize eval2000 and rt03 texts by
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    for x in eval2000 rt03; do
        cp data/${x}/text data/${x}/text.org
        paste -d "" \
              <(cut -f 1 -d" " data/${x}/text.org) \
              <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
            | sed -e 's/\s\+/ /g' > data/${x}/text
        # rm data/${x}/text.org
    done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # make a development set
    utils/subset_data_dir.sh --first data/train 4000 data/${train_dev} # 5hr 6min
    n=$(($(wc -l < data/train/segments) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev
    # remove utterances with frequent words
    utils/data/remove_dup_utts.sh 300 data/train_nodev data/${train_set} # 286hr

    # map acronym such as p._h._d. to p h d for train_set& dev_set
    cp data/${train_set}/text data/${train_set}/text.backup
    cp data/${train_dev}/text data/${train_dev}/text.backup
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/${train_set}/text
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/${train_dev}/text
    if [ -n "${FISHER}" ]; then
        cp data/train_fisher/text data/train_fisher/text.backup
        sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/train_fisher/text
    fi

    # combine swbd and fisher texts
    if [ -n "${FISHER}" ]; then
        gzip -c data/${train_set}/text > data/${train_set}/text.gz
        gzip -c data/train_fisher/text > data/train_fisher/text.gz
        zcat data/${train_set}/text.gz data/train_fisher/text.gz > data/${srctexts}
    fi
    
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
