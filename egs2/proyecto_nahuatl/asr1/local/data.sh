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

audio_dir=downloads/wav
preprocess_dir=data

trans_prefix_Zacatlan=
trans_prefix_Tequila=
trans_prefix_Hidalgo=
audio_prefix_Zacatlan=
audio_prefix_Tequila=
audio_prefix_Hidalgo=

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train"
dev_set="dev"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"
    mkdir -p ${preprocess_dir}/{train,dev,test}
    mkdir -p $audio_dir

    # prepare data
    python3 local/data_prep.py \
        --audio_data_path ${audio_dir} \
        --data_path ${preprocess_dir} \
        --trans_prefix_Zacatlan ${trans_prefix_Zacatlan} \
        --trans_prefix_Tequila ${trans_prefix_Tequila} \
        --trans_prefix_Hidalgo ${trans_prefix_Hidalgo} \
        --audio_prefix_Zacatlan ${audio_prefix_Zacatlan} \
        --audio_prefix_Tequila ${audio_prefix_Tequila} \
        --audio_prefix_Hidalgo ${audio_prefix_Hidalgo}

    for x in dev train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        # fix data dir
        utils/data/fix_data_dir.sh data/${x}
    done

    # prepare for testing set of three languages
    for lan in Hidalgo Tequila Zacatlan; do
        for f in text wav.scp utt2spk; do
            sort data/test/${lan}/${f} -o data/test/${lan}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/test/${lan}/utt2spk > "data/test/${lan}/spk2utt"
        utils/data/fix_data_dir.sh data/test/${lan}
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
