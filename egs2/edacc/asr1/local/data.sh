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
train_set="dev_train"
valid_set="dev_non_train"
test_set="test"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi


if [ -z "${EDACC}" ]; then
    log "Fill the value of 'EDACC' of db.sh"
    exit 1
fi

partitions="${train_set} ${valid_set} ${test_set}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${EDACC}/edacc_v1.0/README.txt" ]; then
        echo "stage 1: Please download data from https://datashare.ed.ac.uk/handle/10283/4836 and save to ${EDACC}"
    else
        log "stage 1: ${EDACC}/edacc_v1.0/README.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    
    # deal with too large wav file in data folder
    audio_path="${EDACC}/edacc_v1.0/data/EDACC-C30.wav"
    output_dir="${EDACC}/edacc_v1.0/data/segmentation"
    mkdir -p "$output_dir"

    if [ -f "$audio_path" ]; then
        # segment at 1883 second
        ffmpeg -i "$audio_path" -ss 0 -t 1883 "$output_dir/EDACC-C30_P1.wav"
        ffmpeg -i "$audio_path" -ss 1883 -c copy "$output_dir/EDACC-C30_P2.wav"
        
        echo "Audio file successfully split into:"
        echo " - $output_dir/EDACC-C30_P1.wav"
        echo " - $output_dir/EDACC-C30_P2.wav"
    else
        echo "File $audio_path not found. Please check the file path."
        exit 1
    fi

    # prepare the date in Kaldi style, output will be "dev" folder and "test" folder in "data" folder
    python3 local/data_prep.py "${EDACC}/edacc_v1.0" "data" "${output_dir}"

        
    # make training data from dev, as original data has no training data
    utils/subset_data_dir.sh --first data/dev 5000 "data/${train_set}" 
    n=$(($(wc -l < data/dev/segments) - 5000))
    utils/subset_data_dir.sh --last data/dev ${n} "data/${valid_set}"


    # sort the data, and make utt2spk to spk2utt
    for x in ${partitions}; do
        for f in text wav.scp utt2spk segments; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done

    # Validate data
    for x in ${partitions}; do
        utils/validate_data_dir.sh --no-feats "data/${x}" 
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
