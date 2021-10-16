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
num_spk=2

 . utils/parse_options.sh || exit 1;

LIBRIMIX=downloads

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
    fi
    # Not installing matplotlib to avoid conflict with ESPnet
    sed -i -e "s/matplotlib>=3\.1\.3//" $FOLDER/requirements.txt
    pip install -r "$FOLDER"/requirements.txt 
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then
# download data & generate librimix
./local/generate_librimix_sd.sh $LIBRIMIX $FOLDER $fs
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] ; then
# Create Kaldi-style files
fs_int=${fs//k/"000"}
mkdir -p data/
python3 local/prepare_diarization.py \
    --target_dir data/ \
    --source_dir ${LIBRIMIX}/Libri${num_spk}Mix/wav${fs}/max/metadata \
    --rttm_dir ${FOLDER}/metadata/LibriSpeech \
    --fs ${fs_int} \
    --num_spk ${num_spk}

for dir in data/test data/train data/dev; do
    utils/fix_data_dir.sh $dir
done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"