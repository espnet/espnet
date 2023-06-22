#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

shopt -s extglob

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
langs="ar as br ca cnh cs cv cy de dv el en eo es et eu\
 fa fr fy-NL ga-IE hsb ia id it ja ka kab ky lv mn mt\
 nl or pa-IN pl pt rm-sursilv rm-vallader ro ru rw sah\
 sl sv-SE ta tr tt uk vi zh-CN zh-HK zh-TW"
lid=true
nlsyms_txt=data/local/nlsyms.txt


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. utils/parse_options.sh

langs=$(echo "${langs}" | tr _ " ")
voxforge_lang="de en es fr it nl pt ru"

train_set=train_li52_lid
train_dev=dev_li52_lid
test_set=

log "data preparation started"

mkdir -p ${COMMONVOICE}
mkdir -p ${VOXFORGE}

for lang in ${langs}; do

    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        log "sub-stage 0: Download Data to ${COMMONVOICE}"

        # base url for downloads.
        # Deprecated url:https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/$lang.tar.gz
        data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/${lang}.tar.gz

        local/download_and_untar.sh ${COMMONVOICE} ${data_url} ${lang}.tar.gz
        rm -f ${COMMONVOICE}/${lang}.tar.gz
    fi

    train_subset=train_"$(echo "${lang}" | tr - _)"_commonvoice
    train_subdev=dev_"$(echo "${lang}" | tr - _)"_commonvoice
    test_subset=test_"$(echo "${lang}" | tr - _)"_commonvoice

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "sub-stage 1: Preparing Data for Commonvoice"

        for part in "validated" "test" "dev"; do
            # use underscore-separated names in data directories.
            local/data_prep.pl "${COMMONVOICE}/cv-corpus-5.1-2020-06-22/${lang}" ${part} data/"$(echo "${part}_${lang}_commonvoice" | tr - _)" "${lang}_commonvoice"
        done

        # remove test&dev data from validated sentences
        utils/copy_data_dir.sh data/"$(echo "validated_${lang}_commonvoice" | tr - _)" data/${train_subset}
        utils/filter_scp.pl --exclude data/${train_subdev}/wav.scp data/${train_subset}/wav.scp > data/${train_subset}/temp_wav.scp
        utils/filter_scp.pl --exclude data/${test_subset}/wav.scp data/${train_subset}/temp_wav.scp > data/${train_subset}/wav.scp
        utils/fix_data_dir.sh data/${train_subset}
    fi
    test_set="${test_set} ${test_subset}"


    if [[ "${voxforge_lang}" == *"${lang}"* ]]; then
        if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
            log "sub-stage0: Download data to ${VOXFORGE}"

            if [ ! -e "${VOXFORGE}/${lang}/extracted" ]; then
                log "sub-stage 1: Download data to ${VOXFORGE}"
                local/getdata.sh "${lang}" "${VOXFORGE}"
            else
                log "sub-stage 1: ${VOXFORGE}/${lang}/extracted is already existing. Skip data downloading"
            fi
        fi

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            log "sub-stage 1: Data Preparation for Voxforge"
            selected=${VOXFORGE}/${lang}/extracted
            # Initial normalization of the data
            local/voxforge_data_prep.sh --flac2wav false "${selected}" "${lang}"
            local/voxforge_format_data.sh "${lang}"
	    utils/copy_data_dir.sh --utt-suffix -${lang}_voxforge data/all_"${lang}" data/validated_"${lang}"_voxforge
	    rm -r data/all_${lang}
            # following split consider prompt duplication (but does not consider speaker overlap instead)
            local/split_tr_dt_et.sh data/validated_"${lang}"_voxforge data/train_"${lang}"_voxforge data/dev_"${lang}"_voxforge data/test_"${lang}"_voxforge
        fi

        test_set="${test_set} test_${lang}_voxforge"

    fi

done

log "Using test sets: ${test_set}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Combine Datadir"

    utils/combine_data.sh --skip_fix true data/train_temp data/train_!(*temp|*li52_*)
    utils/combine_data.sh --skip_fix true data/dev_temp data/dev_!(*temp|*li52_*)

    # Perform text preprocessing (upper case, remove punctuation)
    # Original text:
    #     But, most important, he was able every day to live out his dream.
    #     "Ask me why; I know why."
    # --->
    # Upper text:
    #     BUT, MOST IMPORTANT, HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM.
    #     "ASK ME WHY; I KNOW WHY."
    # ---->
    # Punctuation remove:
    #     BUT MOST IMPORTANT HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM
    #     ASK ME WHY I KNOW WHY

    for x in data/train_temp data/dev_temp; do
        cp ${x}/text ${x}/text.org
        paste -d " " \
              <(cut -f 1 -d" " ${x}/text.org) \
              <(cut -f 2- -d" " ${x}/text.org \
                | python3 -c 'import sys; print(sys.stdin.read().upper(), end="")' \
                | python3 -c 'import string; print(sys.stdin.read().translate(str.maketrans("", "", string.punctuation)), end="")') \
              > ${x}/text
        rm ${x}/text.org
    done

    for x in ${test_set}; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " \
              <(cut -f 1 -d" " data/${x}/text.org) \
              <(cut -f 2- -d" " data/${x}/text.org \
                | python3 -c 'import sys; print(sys.stdin.read().upper(), end="")' \
                | python3 -c 'import string; print(sys.stdin.read().translate(str.maketrans("", "", string.punctuation)), end="")') \
              > data/${x}/text
        rm data/${x}/text.org
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Add Language ID"

    cp -r data/train_temp data/${train_set}
    cp -r data/dev_temp data/${train_dev}

    if [ "$lid" = true ]
    then

        # Original text:
        #     BUT MOST IMPORTANT HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM
        #     ASK ME WHY I KNOW WHY
        # --->
        # Add language ID:
        #     [en] BUT MOST IMPORTANT HE WAS ABLE EVERY DAY TO LIVE OUT HIS DREAM
        #     [en] ASK ME WHY I KNOW WHY

        paste -d " " \
       <(cut -f 1 -d" " data/train_temp/text) \
       <(cut -f 1 -d" " data/train_temp/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
       <(cut -f 2- -d" " data/train_temp/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
       > data/${train_set}/text
        paste -d " " \
       <(cut -f 1 -d" " data/dev_temp/text) \
       <(cut -f 1 -d" " data/dev_temp/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
       <(cut -f 2- -d" " data/dev_temp/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
       > data/${train_dev}/text

        new_test_set=""
        for x in ${test_set}; do
            cp -r data/${x} data/${x}_lid
           paste -d " " \
           <(cut -f 1 -d" " data/${x}/text) \
           <(cut -f 1 -d" " data/${x}/text | sed -e "s/.*\-\(.*\)_.*/\1/" | sed -e "s/_[^TW]\+//" | sed -e "s/^/\[/" -e "s/$/\]/") \
           <(cut -f 2- -d" " data/${x}/text) | sed -e "s/\([^[]*\[[^]]*\]\)\s\(.*\)/\1\2/" \
           > data/${x}_lid/text
           new_test_set="${new_test_set} ${x}_lid"
        done
        echo "test set are saved as ${new_test_set}"

    fi

    utils/fix_data_dir.sh data/${train_set}
    utils/fix_data_dir.sh data/${train_dev}

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Create Non-linguistic Symbols for Language ID"
    cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms_txt}
    log "save non-linguistic symbols in ${nlsyms_txt}"
fi



log "Successfully finished. [elapsed=${SECONDS}s]"
