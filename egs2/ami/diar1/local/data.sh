#!/usr/bin/env bash
# The initial version of this file is copied from librimix/diar1/local/data.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Stage control variables
stage=0       # Start from 0 if you need to start from data preparation
stop_stage=100

# Directory for AMI diarization setup
setup_dir=ami_diarization_setup # Previous name was FOLDER

# Microphone type
# Options: 
# - ihm (individual headset mic) 
# - sdm (single distant mic)
mic_type=ihm 

# Mini dataset flag
# If true, download ami_${data_type}_mini, a subset of the full dataset
if_mini=false

# Specify the type of sounds to be annotated in the RTTM files
# Options:
# - only_words: Annotate only spoken words.
# - word_and_vocalsounds: Annotate both spoken words and vocal sounds (e.g., laughter, coughing).
#                         Note: This could only be used when 'mic_type' is 'ihm'.
# - None: if use mini dataset, the sound_type should be None
# Default is only_words, as vocal sounds are subjectively labeled.
sound_type=only_words

num_spk=4

. utils/parse_options.sh || exit 1;

if [ -z "${AMI}" ]; then
    log "Fill the value of 'AMI' of db.sh"
    exit 1
fi
mkdir -p ${AMI}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

# AMI corpus for speaker diarization setup from gihub : 
#   https://github.com/pyannote/AMI-diarization-setup/tree/main
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] ; then
    # This is a fork created by Qingzheng Wang, mainly modified the database.yml, 
    # to adapt to ESPNet's directory setting
    URL=https://github.com/Qingzheng-Wang/AMI-diarization-setup.git
    # our fork
    if [ ! -d "$setup_dir" ] ; then
        git clone "$URL" "$setup_dir"
        log "Git successfully downloaded AMI-diarization-setup"
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then
    # download data to `downloads`, the downloaded data should be 
    # specified with `mic_type` and `if_mini`, default is `ihm` and `false`.
    if [ ${mic_type} == "ihm" ]; then
        if [ ${if_mini} == false ]; then
            chmod +x ./${setup_dir}/pyannote/download_ami.sh
            ./${setup_dir}/pyannote/download_ami.sh ${AMI}
        else
            chmod +x ./${setup_dir}/pyannote/download_ami_mini.sh
            ./${setup_dir}/pyannote/download_ami_mini.sh ${AMI}
        fi
    elif [ ${mic_type} == "sdm" ]; then 
        if [ ${if_mini} == false ]; then 
            chmod +x ./${setup_dir}/pyannote/download_ami_sdm.sh
            ./${setup_dir}/pyannote/download_ami_sdm.sh ${AMI}
        else
            chmod +x ./${setup_dir}/pyannote/download_ami_sdm_mini.sh
            ./${setup_dir}/pyannote/download_ami_sdm_mini.sh ${AMI}
        fi
    else
        log "mic_type should be 'ihm' or 'sdm', but got ${mic_type}"
        exit 1
    fi

    log "AMI data with mic_type ${mic_type} and if_mini ${if_mini} successfully downloaded"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] ; then
    # Split wav files to short segments, and generate corresponding RTTM files. 
    mkdir -p segmented_dataset/

    log "Start segmenting the dataset"
    python3 local/segment_wav_rttm.py \
        --ami_diarization_config ./${setup_dir}/pyannote/database.yml \
        --mic_type "${mic_type}" \
        --if_mini ${if_mini} \
        --sound_type ${sound_type} \
        --segment_output_dir ./segmented_dataset \
        --duration 20
    
    log "Successfully segmented the dataset"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] ; then
    # Create Kaldi-style files
    mkdir -p data/

    python3 local/prepare_kaldi_files.py \
        --ami_diarization_config ./${setup_dir}/pyannote/database.yml \
        --mic_type "${mic_type}" \
        --if_mini ${if_mini} \
        --sound_type ${sound_type} \
        --kaldi_files_base_dir ./data \
        --num_spk ${num_spk} \
        --segmented_dataset_dir ./segmented_dataset

    # converts the utt2spk file to spk2utt file
    for dir in data/test data/train data/dev; do
        utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
        sort $dir/utt2spk -o $dir/utt2spk
    done

    for dir in data/test data/train data/dev; do
        utils/fix_data_dir.sh $dir
    done

    log "Successfully prepared Kaldi-style files"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
