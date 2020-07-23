#!/bin/bash

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ ! -f path.sh ] || [ ! -f cmd.sh ]; then
    echo "Please change current directory to recipe directory e.g., egs/tedlium2/asr1"
    exit 1
fi

. ./path.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
verbose=1      # verbose option

# feature configuration
do_delta=false
cmvn=

# decoding parameter
align_model=
align_config=
align_dir=align
api=v1

# download related
models=tedlium2.transformer.v1
dict=
nlsyms=

help_message=$(cat <<EOF
Usage:
    $0 [options] <wav_file> "<text>"

Options:
    --backend <chainer|pytorch>     # chainer or pytorch (Default: pytorch)
    --ngpu <ngpu>                   # Number of GPUs (Default: 0)
    --align_dir <directory_name>   # Name of directory to store decoding temporary data
    --models <model_name>           # Model name (e.g. tedlium2.transformer.v1)
    --cmvn <path>                   # Location of cmvn.ark
    --align_model <path>            # Location of E2E model
    --align_config <path>           # Location of configuration file
    --api <api_version>             # API version (v1 or v2, available in only pytorch backend)
    --nlsyms <path>                 # Non-linguistic symbol list

Example:
    # Record audio from microphone input as example.wav
    rec -c 1 -r 16000 example.wav trim 0 5

    # Align using model name
    $0 --models tedlium2.transformer.v1 example.wav "example text"

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
    - wsj.transformer.v1
    - wsj.transformer_small.v1
EOF
)
. utils/parse_options.sh || exit 1;

# make shellcheck happy
train_cmd=
decode_cmd=

. ./cmd.sh

wav=$1
text=$2
download_dir=${align_dir}/download

if [ ! $# -eq 2 ]; then
    echo "${help_message}"
    exit 1;
fi

set -e
set -u
set -o pipefail

# check api version
if [ "${api}" = "v2" ] && [ "${backend}" = "chainer" ]; then
    echo "chainer backend does not support api v2." >&2
    exit 1;
fi

# Check model name or model file is set
if [ -z $models ]; then
    if [[ -z $cmvn || -z $align_model || -z $align_config ]]; then
        echo 'Error: models or set of cmvn, align_model and align_config are required.' >&2
        exit 1
    fi
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
        "librispeech.transformer.v1.transformerlm.v1") share_url="https://drive.google.com/open?id=17cOOSHHMKI82e1MXj4r2ig8gpGCRmG2p" ;;
        "commonvoice.transformer.v1") share_url="https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh" ;;
        "csj.transformer.v1") share_url="https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF" ;;
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
    dict=$(find ${download_dir}/${models}/data/lang_*char -name "*.txt" | head -n 1) || \
        (echo Error: Dictionary file could not be found. Please construct one by yourself following the egs/*/asr1/run.sh. && exit 1;)
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
if [ ! -n "${text}" ]; then
    echo "Text is empty: ${text}"
    exit 1
fi

base=$(basename $wav .wav)
align_dir=${align_dir}/${base}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${align_dir}/data
    echo "$base $wav" > ${align_dir}/data/wav.scp
    echo "X $base" > ${align_dir}/data/spk2utt
    echo "$base X" > ${align_dir}/data/utt2spk
    echo "$base $text" > ${align_dir}/data/text
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
        ${align_dir}/data ${align_dir}/log ${align_dir}/fbank || exit 1;

    feat_align_dir=${align_dir}/dump; mkdir -p ${feat_align_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        ${align_dir}/data/feats.scp ${cmvn} ${align_dir}/log \
        ${feat_align_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    nlsyms_opts=""
    if [ ! -z ${nlsyms} ]; then
        nlsyms_opts="--nlsyms ${nlsyms}"
    fi

    feat_align_dir=${align_dir}/dump
    data2json.sh --feat ${feat_align_dir}/feats.scp ${nlsyms_opts} \
        ${align_dir}/data ${dict} > ${feat_align_dir}/data.json || exit 1;

    unk_id=$(grep "<unk>" ${dict} | awk '{print $2}')
    n_unks=$(grep tokenid ${feat_align_dir}/data.json | \
                sed -e 's/.*: "\(.*\)".*/\1/' | \
                awk -v unk_id=${unk_id} '
                    BEGIN{cnt=0} 
                    {for (i=1;i<=NF;i++) {if ($i==unk_id) {cnt+=1}}} 
                    END{print cnt}
                '
            )
    if [ ${n_unks} -gt 0 ]; then
        echo "Warning: OOVs in the transcriptions could not be aligned."
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Aligning"
    align_opts=""
    feat_align_dir=${align_dir}/dump

    ${decode_cmd} ${align_dir}/log/align.log \
        asr_ctc_align.py \
        --config ${align_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --debugmode ${debugmode} \
        --verbose ${verbose} \
        --align-json ${feat_align_dir}/data.json \
        --result-label ${align_dir}/result.json \
        --model ${align_model} \
        --api ${api} \
        ${align_opts} || exit 1;

    echo ""
    alignment=$(grep ctc_alignment ${align_dir}/result.json | sed -e 's/.*: "\(.*\)".*/\1/' | sed -e 's/<eos>//')
    echo "Alignment: ${alignment}"
    echo ""
    echo "Finished"
fi
