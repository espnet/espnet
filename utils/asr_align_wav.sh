#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
# 2020 Technische Universität München, Authors: Ludwig Kürzinger, Dominik Winkelbauer
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ ! -f path.sh ] || [ ! -f cmd.sh ]; then
    echo "Please change current directory to recipe directory e.g., egs/tedlium2/asr1"
    exit 1
fi

. ./path.sh

# general configuration
python=python3
backend=pytorch
stage=-1       # start from -1 if you need to start from model download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
verbose=1      # verbose option

# feature configuration
do_delta=false
cmvn=

# decoding parameter
align_model=
align_config=
align_dir=align
api=v1

# Parameters for CTC alignment
# The subsampling factor depends on whether the encoder uses subsampling
subsampling_factor=4
# minium confidence score in log space - may need adjustment depending on data and model, e.g. -1.5 or -5.0
min_confidence_score=-5.0
# minimum length of one utterance (counted in frames)
min_window_size=8000
# partitioning length L for calculation of the confidence score
scoring_length=30


# download related
models=tedlium2.rnn.v2
dict=
nlsyms=
download_dir=${align_dir}/download

. utils/parse_options.sh || exit 1;

help_message=$(cat <<EOF
Usage:
    $0 [options] <wav_file> "<text>"
    $0 [options] <wav_file> <utt_text_file>

Options:
    --backend <chainer|pytorch>     # chainer or pytorch (Default: pytorch)
    --ngpu <ngpu>                   # Number of GPUs (Default: 0)
    --align-dir <directory_name>    # Name of directory to store decoding temporary data
    --download-dir <directory_name> # Name of directory to store download files
    --models <model_name>           # Model name (e.g. tedlium2.transformer.v1)
    --cmvn <path>                   # Location of cmvn.ark
    --align-model <path>            # Location of E2E model
    --align-config <path>           # Location of configuration file
    --api <api_version>             # API version (v1 or v2, available in only pytorch backend)
    --nlsyms <path>                 # Non-linguistic symbol list

Example:
    # Record audio from microphone input as example.wav
    rec -c 1 -r 16000 example.wav trim 0 5

    # Align using model name
    $0 --models tedlium2.transformer.v1 example.wav "example text"

    $0 --models tedlium2.transformer.v1 example.wav utt_text.txt

    # Align using model file
    $0 --cmvn cmvn.ark --align_model model.acc.best --align_config conf/align.yaml example.wav

    # Align with GPU (require batchsize > 0 in configuration file)
    $0 --ngpu 1 example.wav

Available models:
    - tedlium2.rnn.v1
    - tedlium2.rnn.v2
    - tedlium2.transformer.v1
    - tedlium3.transformer.v1
    - librispeech.transformer.v1
    - librispeech.transformer.v1.transformerlm.v1
    - commonvoice.transformer.v1
    - csj.transformer.v1
    - csj.rnn.v1
    - wsj.transformer.v1
    - wsj.transformer_small.v1
EOF
)


# make shellcheck happy
train_cmd=

. ./cmd.sh

wav=$1
text=$2

if [ ! $# -eq 2 ]; then
    echo "${help_message}"
    exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check api version
if [ "${backend}" = "chainer" ]; then
    echo "chainer backend is not supported." >&2
    exit 1;
fi

# Check model name or model file is set
if [ -z $models ]; then
    if [[ -z $cmvn || -z $align_model || -z $align_config ]]; then
        echo 'Error: models or set of cmvn, align_model and align_config are required.' >&2
        exit 1
    fi
fi

# Check for transformer models because of their memory consumption
if [[ $models == *"rnn"* ]]; then
    echo "Using RNN model: "${models}
else
    echo "Using Transformer model: "${models}
    echo "WARNING. For large audio files, use an RNN model."
fi

dir=${download_dir}/${models}
mkdir -p ${dir}

function download_models () {
    if [ -z $models ]; then
        return
    fi

    file_ext="tar.gz"
    case "${models}" in
        "tedlium2.rnn.v1") share_url="https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe"; api=v1 ;;
        "tedlium2.rnn.v2") share_url="https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf"; api=v1 ;;
        "tedlium2.transformer.v1") share_url="https://drive.google.com/open?id=1cVeSOYY1twOfL9Gns7Z3ZDnkrJqNwPow" ;;
        "tedlium3.transformer.v1") share_url="https://drive.google.com/open?id=1zcPglHAKILwVgfACoMWWERiyIquzSYuU" ;;
        "librispeech.transformer.v1") share_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" ;;
        "librispeech.transformer.v1.transformerlm.v1") share_url="https://drive.google.com/open?id=1RHYAhcnlKz08amATrf0ZOWFLzoQphtoc" ;;
        "commonvoice.transformer.v1") share_url="https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh" ;;
        "csj.transformer.v1") share_url="https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF" ;;
        "csj.rnn.v1") share_url="https://drive.google.com/open?id=1ALvD4nHan9VDJlYJwNurVr7H7OV0j2X9" ;;
        "wsj.transformer.v1") share_url="https://drive.google.com/open?id=1Az-4H25uwnEFa4lENc-EKiPaWXaijcJp" ;;
        "wsj.transformer_small.v1") share_url="https://drive.google.com/open?id=1jdEKbgWhLTxN_qP4xwE7mTOPmp7Ga--T" ;;
        *) echo "No such models: ${models}"; exit 1 ;;
    esac

    if [ ! -e ${dir}/.complete ]; then
        download_from_google_drive.sh ${share_url} ${dir} ${file_ext}
        touch ${dir}/.complete
    fi
}

# Download trained models
if [ -z "${cmvn}" ]; then
    download_models
    cmvn=$(find ${download_dir}/${models} -name "cmvn.ark" | head -n 1)
fi
if [ -z "${align_model}" ]; then
    download_models
    align_model=$(find ${download_dir}/${models} -name "model*.best*" | head -n 1)
fi
if [ -z "${align_config}" ]; then
    download_models
    align_config=$(find ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi
if [ -z "${wav}" ]; then
    download_models
    wav=$(find ${download_dir}/${models} -name "*.wav" | head -n 1)
fi
if [ -z "${dict}" ]; then
    download_models

    if [ -z "${dict}" ]; then
        mkdir -p ${download_dir}/${models}/data/lang_autochar/
        model_config=$(find -L ${download_dir}/${models}/exp/*/results/model.json | head -n 1)
        dict=${download_dir}/${models}/data/lang_autochar/dict.txt
        python -c 'import json,sys;obj=json.load(sys.stdin);[print(char + " " + str(i + 1)) for i, char in enumerate(obj[2]["char_list"])]' > ${dict} < ${model_config}
    fi
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${align_model}" ]; then
    echo "No such E2E model: ${align_model}"
    exit 1
fi
if [ ! -f "${align_config}" ]; then
    echo "No such config file: ${align_config}"
    exit 1
fi
if [ ! -f "${dict}" ]; then
    echo "No such Dictionary file: ${dict}"
    exit 1
fi
if [ ! -f "${wav}" ]; then
    echo "No such WAV file: ${wav}"
    exit 1
fi
if [ -z "${text}" ]; then
    echo "Text is empty: ${text}"
    exit 1
fi

base=$(basename $wav .wav)

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${align_dir}/data
    echo "$base $wav" > ${align_dir}/data/wav.scp
    echo "X $base" > ${align_dir}/data/spk2utt
    echo "$base X" > ${align_dir}/data/utt2spk
    utt_text="${align_dir}/data/text"
    if [ -f "$text" ]; then
        cp -v "$text" "$utt_text"
        utt_text="${text}" # Use the original file, because copied file will be truncated
    else
        echo "$base $text" > "${utt_text}"
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
        ${align_dir}/data ${align_dir}/log ${align_dir}/fbank || exit 1;

    feat_align_dir=${align_dir}/dump; mkdir -p ${feat_align_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        ${align_dir}/data/feats.scp ${cmvn} ${align_dir}/log \
        ${feat_align_dir}
    utils/fix_data_dir.sh ${align_dir}/data
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    nlsyms_opts=""
    if [[ -n ${nlsyms} ]]; then
        nlsyms_opts="--nlsyms ${nlsyms}"
    fi

    feat_align_dir=${align_dir}/dump
    data2json.sh --feat ${feat_align_dir}/feats.scp ${nlsyms_opts} \
        ${align_dir}/data ${dict} > ${feat_align_dir}/data.json || exit 1;

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Aligning"
    feat_align_dir=${align_dir}/dump

    ${python} -m espnet.bin.asr_align \
        --config ${align_config} \
        --ngpu ${ngpu} \
        --verbose ${verbose} \
        --data-json ${feat_align_dir}/data.json \
        --model ${align_model} \
        --subsampling-factor ${subsampling_factor} \
        --min-window-size ${min_window_size} \
        --scoring-length ${scoring_length} \
        --api ${api} \
        --utt-text ${utt_text} \
        --output ${align_dir}/aligned_segments || exit 1;

    echo ""
    echo "Segments file: $(wc -l ${align_dir}/aligned_segments)"
    count_reliable=$(awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' ${align_dir}/aligned_segments | wc -l)
    echo "Utterances with min confidence score: ${count_reliable}"
    echo "Finished."
fi
