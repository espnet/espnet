#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
nlsyms_txt=data/local/nlsyms.txt
duration=10min # duration can be either 10min or 1h
multilingual=true
lid=false
only_lid=false
single_lang=eng # lang for single lang data preparation
                # candidates: eng, deu, rus, pol, swe, jpn, cmn, sat, nob, xty

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MLSUPERB}
if [ -z "${MLSUPERB}" ]; then
    log "Fill the value of 'MLSUPERB' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Download data to ${MLSUPERB}"
    log "Please use the download link in readme and set the MLSUPERB as its unzipped path."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage2: Preparing data for multilingual SUPERB"

    if "${multilingual}"; then
        if "${only_lid}"; then
            suffix="_only_lid"
        else
            if "${lid}"; then
                suffix="_lid"
            else
                suffix=""
            fi
        fi
        mkdir -p data/train_${duration}${suffix}
        mkdir -p data/dev_${duration}${suffix}
        mkdir -p data/test_${duration}${suffix}

        python local/data_prep.py \
            --train_set train_${duration}${suffix} \
            --train_dev dev_${duration}${suffix} \
            --test_set test_${duration}${suffix} \
            --duration ${duration} \
            --source ${MLSUPERB} \
            --lid ${lid} \
            --only_lid ${only_lid}

        for x in "train" "dev" "test"; do
            utils/utt2spk_to_spk2utt.pl \
                data/${x}_${duration}${suffix}/utt2spk \
                > data/${x}_${duration}${suffix}/spk2utt
            utils/fix_data_dir.sh data/${x}_${duration}${suffix}
        done
    else
        for x in "train" "dev" "test"; do
            mkdir -p data/${x}_${duration}_${single_lang}
        done

        python local/single_lang_data_prep.py \
            --duration ${duration} \
            --source ${MLSUPERB} \
            --lang ${single_lang}

        for x in "train" "dev" "test"; do
             utils/utt2spk_to_spk2utt.pl \
                 data/${x}_${duration}_${single_lang}/utt2spk \
                 > data/${x}_${duration}_${single_lang}/spk2utt

            if [ "${single_lang}" == "cmn" ]; then
                g2p=pypinyin_g2p_phone
            elif [ "${single_lang}" == "jpn" ]; then
                g2p=pyopenjtalk
                # check extra module installation
                if ! python3 -c "import pyopenjtalk" > /dev/null; then
                    echo "Error: pyopenjtalk is not installed (but need for jpn)." >&2
                    echo "Installing with ESPnet Makefile"
                    msuperb_dir=$(pwd)
                    cd "${MAIN_ROOT}"/tools && make pyopenjtalk.done
                    cd "${msuperb_dir}"
                fi
            else
                g2p=none
            fi

            utils/fix_data_dir.sh data/${x}_${duration}_${single_lang}

            if [ "${single_lang}" == "cmn" ] || [ "${single_lang}" == "jpn" ]; then
                python -m espnet2.bin.tokenize_text --token_type "phn" -f 2- \
                    --input "data/${x}_${duration}_${single_lang}/text"    \
                    --output "data/${x}_${duration}_${single_lang}/phn_text" \
                    --cleaner "none" \
                    --g2p "${g2p}"
                paste -d " " <(cut -f1 -d" " data/${x}_${duration}_${single_lang}/wav.scp) \
                    <(cat data/${x}_${duration}_${single_lang}/phn_text) \
                    > data/${x}_${duration}_${single_lang}/text
                utils/fix_data_dir.sh data/${x}_${duration}_${single_lang}
            fi


        done
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage3: Create non-linguistic symbols for language ID"
    mkdir -p "$(dirname ${nlsyms_txt})"
    if "${multilingual}" && "${lid}"; then
        train_set=data/train_${duration}${suffix}
        cut -f 2- ${train_set}/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms_txt}
        log "save non-linguistic symbols in ${nlsyms_txt}"
    else
        touch ${nlsyms_txt}
        log "no non-linguistic symbols needed"
    fi
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
