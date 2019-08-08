#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ ! -f path.sh ] || [ ! -f cmd.sh ]; then
    echo "Please change directory to e.g., egs/ljspeech/tts1"
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
fs=22050      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length
cmvn=

# dictionary related
dict=

# embedding related
input_wav=

# decoding related
synth_model=
decode_config=
decode_dir=decode
griffin_lim_iters=64

# download related
models=ljspeech.transformer.v1

help_message=$(cat <<EOF
Usage:
    $ $0 <text>

Example:
    # make text file and then generate it
    echo "This is a demonstration of text to speech." > example.txt
    $0 example.txt

    # you can specify the pretrained models
    $0 --models ljspeech.tacotron2.v3 example.txt

Available models:
    - libritts.tacotron2.v1
    - ljspeech.tacotron2.v1
    - ljspeech.tacotron2.v2
    - ljspeech.tacotron2.v3
    - ljspeech.transformer.v1
    - ljspeech.transformer.v2
    - ljspeech.fastspeech.v1
    - ljspeech.fastspeech.v2
EOF
)
. utils/parse_options.sh || exit 1;

# make shellcheck happy
train_cmd=
decode_cmd=

. ./cmd.sh

txt=$1
download_dir=${decode_dir}/download

if [ $# -ne 1 ]; then
    echo "${help_message}"
    exit 1;
fi

set -e
set -u
set -o pipefail

function download_models () {
    case "${models}" in
        "libritts.tacotron2.v1") share_url="https://drive.google.com/open?id=1iAXwC0AuWusa9AcFeUVkcNLG0I-hnSr3" ;;
        "ljspeech.tacotron2.v1") share_url="https://drive.google.com/open?id=1dKzdaDpOkpx7kWZnvrvx2De7eZEdPHZs" ;;
        "ljspeech.tacotron2.v2") share_url="https://drive.google.com/open?id=11T9qw8rJlYzUdXvFjkjQjYrp3iGfQ15h" ;;
        "ljspeech.tacotron2.v3") share_url="https://drive.google.com/open?id=1hiZn14ITUDM1nkn-GkaN_M3oaTOUcn1n" ;;
        "ljspeech.transformer.v1") share_url="https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS" ;;
        "ljspeech.transformer.v2") share_url="https://drive.google.com/open?id=1xxAwPuUph23RnlC5gym7qDM02ZCW9Unp" ;;
        "ljspeech.fastspeech.v1") share_url="https://drive.google.com/open?id=1BAkmFcG5QdBgw4onr6BGJb3cDVDcdKRA" ;;
        "ljspeech.fastspeech.v2") share_url="https://drive.google.com/open?id=1wEOJjkajJCet7HzzGV5igPcwfh2n-yOK";;
        *) echo "No such models: ${models}"; exit 1 ;;
    esac

    dir=${download_dir}/${models}
    mkdir -p ${dir}
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
if [ -z "${dict}" ]; then
    download_models
    dict=$(find ${download_dir}/${models} -name "*_units.txt" | head -n 1)
fi
if [ -z "${synth_model}" ]; then
    download_models
    synth_model=$(find ${download_dir}/${models} -name "model*.best" | head -n 1)
fi
if [ -z "${decode_config}" ]; then
    download_models
    decode_config=$(find ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi

synth_json=$(basename ${synth_model})
model_json="$(dirname ${synth_model})/${synth_json%%.*}.json"
use_speaker_embedding=$(grep use_speaker_embedding ${model_json} | sed -e "s/.*: \(.*\),/\1/")
if [ "${use_speaker_embedding}" = "false" ] || [ "${use_speaker_embedding}" = "0" ]; then
    use_input_wav=false
else
    use_input_wav=true
fi
if [ -z "${input_wav}" ] && ${use_input_wav}; then
    download_models
    input_wav=$(find ${download_dir}/${models} -name "*.wav" | head -n 1)
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${dict}" ]; then
    echo "No such dictionary: ${dict}"
    exit 1
fi
if [ ! -f "${synth_model}" ]; then
    echo "No such E2E model: ${synth_model}"
    exit 1
fi
if [ ! -f "${decode_config}" ]; then
    echo "No such config file: ${decode_config}"
    exit 1
fi
if [ ! -f "${input_wav}" ] && ${use_input_wav}; then
    echo "No such WAV file for extracting meta information: ${input_wav}"
    exit 1
fi
if [ ! -f "${txt}" ]; then
    echo "No such txt file: ${txt}"
    exit 1
fi

base=$(basename $txt .txt)
decode_dir=${decode_dir}/${base}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${decode_dir}/data
    echo "$base X" > ${decode_dir}/data/wav.scp
    echo "X $base" > ${decode_dir}/data/spk2utt
    echo "$base X" > ${decode_dir}/data/utt2spk
    echo -n "$base " > ${decode_dir}/data/text
    cat $txt >> ${decode_dir}/data/text

    mkdir -p ${decode_dir}/dump
    data2json.sh ${decode_dir}/data ${dict} > ${decode_dir}/dump/data.json
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ${use_input_wav}; then
    echo "stage 1: x-vector extraction"

    utils/copy_data_dir.sh ${decode_dir}/data ${decode_dir}/data2
    echo "$base ${input_wav}" > ${decode_dir}/data2/wav.scp
    utils/data/resample_data_dir.sh 16000 ${decode_dir}/data2
    steps/make_mfcc.sh \
        --write-utt2num-frames true \
        --mfcc-config conf/mfcc.conf \
        --nj 1 --cmd "$train_cmd" \
        ${decode_dir}/data2 ${decode_dir}/log ${decode_dir}/mfcc
    utils/fix_data_dir.sh ${decode_dir}/data2
    sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
        ${decode_dir}/data2 ${decode_dir}/log ${decode_dir}/mfcc
    utils/fix_data_dir.sh ${decode_dir}/data2

    nnet_dir=${download_dir}/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a ${download_dir}
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
        ${nnet_dir} ${decode_dir}/data2 \
        ${decode_dir}/xvectors

    local/update_json.sh ${decode_dir}/dump/data.json ${decode_dir}/xvectors/xvector.scp
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Decoding"

    ${decode_cmd} ${decode_dir}/log/decode.log \
        tts_decode.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --debugmode ${debugmode} \
        --verbose ${verbose} \
        --out ${decode_dir}/outputs/feats \
        --json ${decode_dir}/dump/data.json \
        --model ${synth_model}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Synthesis"

    outdir=${decode_dir}/outputs; mkdir -p ${outdir}_denorm
    apply-cmvn --norm-vars=true --reverse=true ${cmvn} \
        scp:${outdir}/feats.scp \
        ark,scp:${outdir}_denorm/feats.ark,${outdir}_denorm/feats.scp

    convert_fbank.sh --nj 1 --cmd "${decode_cmd}" \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        --iters ${griffin_lim_iters} \
        ${outdir}_denorm \
        ${decode_dir}/log \
        ${decode_dir}/wav

    echo ""
    echo "Synthesized wav: ${decode_dir}/wav/${base}.wav"
    echo ""
    echo "Finished"
fi
