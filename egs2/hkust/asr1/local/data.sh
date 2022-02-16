#!/usr/bin/env bash
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
stop_stage=100

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh
. ./db.sh


if [ -z "${HKUST1}" ]; then
  log "Error: \$HKUST1 is not set in db.sh."
  exit 2
fi
if [ -z "${HKUST2}" ]; then
  log "Error: \$HKUST2 is not set in db.sh."
  exit 2
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/hkust_data_prep.sh "${HKUST1}" "${HKUST2}"
    local/hkust_format_data.sh

    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train dev; do
        sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    done

    # remove space in text
    for x in train dev; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
    done

    # 4001 utts will be reduced to 4000 after feature extraction
    utils/subset_data_dir.sh --first data/train 4001 data/train_dev
    utils/fix_data_dir.sh data/train_dev
    n=$(($(wc -l < data/train/segments) - 4001))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev

    # make a training set
    utils/data/remove_dup_utts.sh 300 data/train_nodev data/train_nodup


    cut -f 2- data/train/text | grep -o -P '\[.*?\]' | sort | uniq > data/nlsyms.txt
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
