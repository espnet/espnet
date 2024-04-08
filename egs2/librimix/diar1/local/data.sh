#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
FOLDER=git_librimix
fs=8k
num_spk="2 3"
min_max_mode=max

 . utils/parse_options.sh || exit 1;

if [ -z "${LIBRIMIX}" ]; then
    log "Fill the value of 'LIBRIMIX' of db.sh"
    exit 1
fi
mkdir -p ${LIBRIMIX}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

# Github LibriMix : https://github.com/s3prl/LibriMix.git
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] ; then
    URL=https://github.com/YushiUeda/LibriMix.git
    # our fork
    if [ ! -d "$FOLDER" ] ; then
        git clone "$URL" "$FOLDER"
        log "git successfully downloaded"
        # Not installing matplotlib to avoid conflict with ESPnet
        sed -i -e "s/matplotlib>=3\.1\.3//" $FOLDER/requirements.txt
        pip install -r "$FOLDER"/requirements.txt
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then
# download data & generate librimix
./local/generate_librimix_sd.sh $LIBRIMIX $FOLDER $fs $min_max_mode
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] ; then
# Create Kaldi-style files
fs_int=${fs//k/"000"}
mkdir -p data/

for i in $num_spk; do
    python3 local/prepare_kaldifiles.py \
        --target_dir data/ \
        --source_dir ${LIBRIMIX}/Libri${i}Mix/wav${fs}/${min_max_mode}/metadata \
        --rttm_dir ${FOLDER}/metadata/LibriSpeech \
        --fs ${fs_int} \
        --num_spk $i
done

# rm unnecessary files
for i in $num_spk; do
    for dir in data/test data/train data/dev; do
        rm ${dir}${i}/spk*.scp noise1.scp
    done
done

for file in reco2dur rttm segments spk2utt utt2spk wav.scp; do
    for dir in data/test data/train data/dev; do
        mkdir -p ${dir}
        echo -n "" > ${dir}/${file}
        for i in $num_spk; do
            cat ${dir}${i}/${file} >> ${dir}/${file}
        done
    done
done

for dir in data/test data/train data/dev; do
    utils/fix_data_dir.sh $dir
done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
