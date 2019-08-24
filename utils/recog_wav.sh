#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#           2019 RevComm Inc. (Takekatsu Hiramura)
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

# rnnlm related
use_lang_model=true
lang_model=

# decoding parameter
recog_model=
decode_config=
decode_dir=decode
api=v1

# download related
models=

help_message=$(cat <<EOF
Usage:
    $0 [options] <wav_file>

Options:
  --backend <chainer|pytorch>     # chainer or pytorch (Default: pytorch)
  --ngpu <ngpu>                   # Number of GPUs (Default: 0)
  --decode_dir <directory_name>   # Name of directory to store decoding temporary data
  --models <model_name>           # Model name (e.g. tedlium2.tacotron2.v1)
  --cmvn <path>                   # Location of cmvn.ark
  --lang_model <path>             # Location of language model
  --recog_model <path>            # Location of E2E model
  --decode_config <path>          # Location of configuration file
  --api <api_version>             # API version (v1 or v2, available in only pytorch backend)

Example:
    # Record audio from microphone input as example.wav
    rec -c 1 -r 16000 example.wav trim 0 5

    # Decode using model name
    $0 --model tedlium2.tacotron2.v1 example.wav

    # Decode using model file
    $0 --cmvn cmvn.ark --lang_model rnnlm.model.best --recog_model model.acc.best --decode_config conf/decode.yaml example.wav
EOF
)
. utils/parse_options.sh || exit 1;

# make shellcheck happy
train_cmd=
decode_cmd=

. ./cmd.sh

wav=$1
download_dir=${decode_dir}/download

if [ $# -lt 1 ]; then
    echo "${help_message}"
    exit 1;
fi

set -e
set -u
set -o pipefail

# Check model name or model file is set
if [ -z $models ]; then
    if [ $use_lang_model = "true" ]; then
        if [[ -z $cmvn || -z $lang_model || -z $recog_model || -z $decode_config ]]; then
            echo 'Error: models or set of cmvn, lang_model, recog_model and decode_config are required.' >&2
            exit 1
        fi
    else
        if [[ -z $cmvn || -z $recog_model || -z $decode_config ]]; then
            echo 'Error: models or set of cmvn, recog_model and decode_config are required.' >&2
            exit 1
        fi
    fi
fi

dir=${download_dir}/${models}
mkdir -p ${dir}

function download_models () {
    if [ -z $models ]; then
        return
    fi
    case "${models}" in
        "tedlium2.tacotron2.v1") share_url="https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe" ;;
        *) echo "No such models: ${models}"; exit 1 ;;
    esac

    if [ ! -e ${dir}/.complete ]; then
        download_from_google_drive.sh ${share_url} ${dir} ".tar.gz"
        touch ${dir}/.complete
    fi
}

# Download trained models
if [ -z "${cmvn}" ]; then
    download_models
    cmvn=$(find ${download_dir}/${models} -name "cmvn.ark" | head -n 1)
fi
if [ -z "${lang_model}" ] && ${use_lang_model}; then
    download_models
    lang_model=$(find ${download_dir}/${models} -name "rnnlm*.best" | head -n 1)
fi
if [ -z "${recog_model}" ]; then
    download_models
    recog_model=$(find ${download_dir}/${models} -name "model*.best" | head -n 1)
fi
if [ -z "${decode_config}" ]; then
    download_models
    decode_config=$(find ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi
if [ -z "${wav}" ]; then
    download_models
    wav=$(find ${download_dir}/${models} -name "*.wav" | head -n 1)
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${lang_model}" ] && ${use_lang_model}; then
    echo "No such language model: ${lang_model}"
    exit 1
fi
if [ ! -f "${recog_model}" ]; then
    echo "No such E2E model: ${recog_model}"
    exit 1
fi
if [ ! -f "${decode_config}" ]; then
    echo "No such config file: ${decode_config}"
    exit 1
fi
if [ ! -f "${wav}" ]; then
    echo "No such WAV file: ${wav}"
    exit 1
fi

base=$(basename $wav .wav)
decode_dir=${decode_dir}/${base}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${decode_dir}/data
    echo "$base $wav" > ${decode_dir}/data/wav.scp
    echo "X $base" > ${decode_dir}/data/spk2utt
    echo "$base X" > ${decode_dir}/data/utt2spk
    echo "$base X" > ${decode_dir}/data/text
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
        ${decode_dir}/data ${decode_dir}/log ${decode_dir}/fbank

    feat_recog_dir=${decode_dir}/dump; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta ${do_delta} \
        ${decode_dir}/data/feats.scp ${cmvn} ${decode_dir}/log \
        ${feat_recog_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    dict=${decode_dir}/dict
    echo "<unk> 1" > ${dict}
    feat_recog_dir=${decode_dir}/dump
    data2json.sh --feat ${feat_recog_dir}/feats.scp \
        ${decode_dir}/data ${dict} > ${feat_recog_dir}/data.json
    rm -f ${dict}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"

    if ${use_lang_model}; then
        recog_opts="--rnnlm ${lang_model}"
    else
        recog_opts=""
    fi
    feat_recog_dir=${decode_dir}/dump

    ${decode_cmd} ${decode_dir}/log/decode.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --debugmode ${debugmode} \
        --verbose ${verbose} \
        --recog-json ${feat_recog_dir}/data.json \
        --result-label ${decode_dir}/result.json \
        --model ${recog_model} \
        --api ${api} \
        ${recog_opts}

    echo ""
    recog_text=$(grep rec_text ${decode_dir}/result.json | sed -e 's/.*: "\(.*\)".*/\1/' | sed -e 's/<eos>//')
    echo "Recognized text: ${recog_text}"
    echo ""
    echo "Finished"
fi
