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

download_and_untar=false
mp4_to_wav=true

# Manually fill the lrs3_username, lrs3_password
lrs3_username=
lrs3_password=

# Multimodal Related
audio_input=true
vision_input=false
mouth_roi=false

# Read Arguments
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 1
fi

if [ -z "${LRS3}" ]; then
    log "Fill the value of 'LRS3' of db.sh"
    exit 1
fi

if $download_and_untar; then
    log "Downloading and Untarring the LRS3 with username ${lrs3_username} and passwoed ${lrs3_password}."
    local/download_and_untar.sh --remove-archive ${LRS3} ${lrs3_username} ${lrs3_password}
fi

if $mp4_to_wav; then
    log "Extacting .wav files from .mp4 files and storing it under the same directory"
    local/mp4_to_wav.sh ${LRS3}
fi

# Make the Folders where ESPNet data-prep files will be stored
for dataset in train dev test; do
    log "Creating the ./data/${dataset} folders"
    mkdir -p ./data/${dataset}
done

# Multimodal Related: Perform any Preprocessing i.e. Mouth ROI Cropping
log "Preprocessing step for vision data input"
local/vision_preprocess.sh --mouth_roi ${mouth_roi}

# generate the utt2spk, wav.scp and text files
log "Generating the utt2spk, wav.scp and text files"
python3 ./local/data_prep.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test --audio_input ${audio_input} --vision_input ${vision_input} --mouth_roi ${mouth_roi}

log "Generating the spk2utt files"
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

log "Fix sorting issues by calling fix_data_dir.sh"
utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/test
utils/fix_data_dir.sh data/dev

log "Validate the data directory"
utils/validate_data_dir.sh data/train --no-feats
utils/validate_data_dir.sh data/test --no-feats
utils/validate_data_dir.sh data/dev --no-feats
