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

# Multimodal Related
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

if "${mouth_roi}"; then
    log "Extracting mouth_roi.mp4 files and storing it under the same directory"
    if ! [ -f ${face_predictor_path} ]; then
        wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O "${face_predictor_path}.bz2"
        bzip2 -d "${face_predictor_path}.bz2"
    fi
    if ! [ -f ${mean_face_path} ]; then
        wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O ${mean_face_path}
    fi
    python3 ./local/mouth_roi.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test --face_predictor_path ${face_predictor_path} --mean_face_path ${mean_face_path} --ffmpeg_path ${ffmpeg_path}
fi
