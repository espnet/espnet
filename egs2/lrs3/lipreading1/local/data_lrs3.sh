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

. ./db.sh
. ./path.sh
. ./cmd.sh

cmd=run.pl
log_dir=./local/python_debug_log
nj=1

echo ${LRS3}

# for dataset in trainval test; do
#     for mp4_path in ${LRS3}/${dataset}/*/*.mp4; do
#         wav_path=${mp4_path//.mp4/.wav}
#         ffmpeg -y -i ${mp4_path} -loglevel panic -ar 16000 -ac 1 ${wav_path} 
#     done
# done

# Make the Folders where ESPNet essential files will be stored
# for dataset in train dev test; do
#     mkdir -p ./data/${dataset}
# done

# $cmd JOB=1:$nj ${log_dir}.JOB.log python ./local/data_prep.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test 

utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/test
utils/fix_data_dir.sh data/dev

utils/validate_data_dir.sh data/train --no-feats
utils/validate_data_dir.sh data/test --no-feats
utils/validate_data_dir.sh data/dev --no-feats

