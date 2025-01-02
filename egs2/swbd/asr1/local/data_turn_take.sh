#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
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

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SWBD}" ]; then
    log "Fill the value of 'SWBD' of db.sh"
    exit 1
fi


# we assume the following data structure
  # SWBD: LDC97S62 LDC2002S09 LDC2002T43 LDC2004T19 LDC2005T19 LDC2004S13 LDC2005S13
swbd1_dir=${SWBD}
# eval2000_dir="${SWBD}/LDC2002S09/hub5e_00 ${SWBD}/LDC2002T43"
# fisher_dir="${SWBD}/LDC2004T19 ${SWBD}/LDC2005T19 ${SWBD}/LDC2004S13 ${SWBD}/LDC2005S13"



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log " Data Preparation"
    local/swbd1_data_download.sh ${swbd1_dir}
    local/swbd1_prepare_dict.sh
    local/swbd1_data_prep.sh ${swbd1_dir}
    # local/eval2000_data_prep.sh ${eval2000_dir}
    # if [ -n "${fisher_dir}" ]; then
    #      local/fisher_data_prep.sh ${fisher_dir}
    # fi
    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    done

    # x=eval2000
    # cp data/${x}/text data/${x}/text.org
    # paste -d "" \
    #      <(cut -f 1 -d" " data/${x}/text.org) \
    #      <(awk '{$1=""; print tolower($0)}' data/${x}/text.org \
    #      | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' \
    #      | sed -e "s/(//g" -e "s/)//g") \
    #      | sed -e 's/\s\+/ /g' > data/${x}/text.org2 # for ci check
    # # remove the file with empty text, otherwise bug in stage calc perplexity
    # awk -F ' ' '{if(length($2)!=0)print $0}' data/${x}/text.org2 > data/${x}/text


    utils/fix_data_dir.sh data/train
    # utils/fix_data_dir.sh data/eval2000
    # utils/subset_data_dir.sh --first data/train 4000 data/train_dev # 5hr 6min
    # n=$(($(wc -l < data/train/segments) - 4000))
    # utils/subset_data_dir.sh --last data/train ${n} data/train_nodev
    utils/data/remove_dup_utts.sh 300 data/train data/train_nodup # 286hr

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log " Data Formatting"
     # remove ._ . _1 symbols from text
     cp data/train_nodup/text data/train_nodup/text.backup
    #  cp data/train_dev/text data/train_dev/text.backup
     sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/train_nodup/text
    #  sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/train_dev/text
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 2: Data Preparation"
    python local/get_duration.py
    sh sox_duration.sh &> sox_duration.txt
    wget https://raw.githubusercontent.com/ErikEkstedt/VoiceActivityProjection/refs/heads/main/dataset_swb/backchannels.csv -O  local/backchannels.csv
    python local/create_switchboard_data_2channels.py
    python local/create_switchboard_data_2channels_mono.py
    python local/subsample_2channel_switchboard_mono.py
    mkdir -p data/{train,valid,test}
    python3 local/create_espnet_data_folders.py
    for x in test valid train; do
        for f in segments text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
