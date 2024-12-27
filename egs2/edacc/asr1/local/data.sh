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
sub_test_set="test_sub"

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

partitions="${train_set} ${valid_set} ${test_set} ${sub_test_set}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${EDACC}/edacc_v1.0/README.txt" ]; then
        echo "stage 1: Please download data from https://datashare.ed.ac.uk/handle/10283/4836 and save to ${EDACC}"
    else
        log "stage 1: ${EDACC}/edacc_v1.0/README.txt is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation -- preprocess large wav files"

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
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Data preparation -- prepare kaldi files, generate ${train_set}, 
    ${valid_set}, ${test_set}, ${sub_test_set}"

    # prepare the date in Kaldi style, output will be "dev" folder and "test" folder in "data" folder
    python3 local/data_prep.py "${EDACC}/edacc_v1.0" "data" "${output_dir}"

    # # (optional) split the too long test utterance used for decoding section if necessary,
    # # the alignment is based on CTC segmentation tool
    # python3 local/truncate_test.py "data/test"

    # make training data from dev, as original data has no training data
    utils/subset_data_dir.sh --utt-list data/train_utterlist data/dev "data/${train_set}"
    utils/subset_data_dir.sh --utt-list data/valid_utterlist data/dev "data/${valid_set}"

    # make a sub test set from test set
    utils/subset_data_dir.sh --first data/test 500 "data/${sub_test_set}"

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
