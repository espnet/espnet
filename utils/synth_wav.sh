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
fs=24000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length
cmvn=

# dictionary related
dict=

# x-vector related
use_input_wav=false
input_wav="input.wav"

# decoding related
synth_model=
decode_config=
decode_dir=decode
griffin_lim_iters=1000

# download related
models=libritts.v1

. utils/parse_options.sh || exit 1;

# make shellcheck happy
train_cmd=
decode_cmd=

. ./cmd.sh

txt=$1
download_dir=${decode_dir}/download

if [ $# -gt 1 ]; then
    echo "Usage: $0 <wav>"
    exit 1;
fi

set -e
set -u
set -o pipefail

case "${models}" in
    "libritts.v1")
        share_url="https://drive.google.com/open?id=1iAXwC0AuWusa9AcFeUVkcNLG0I-hnSr3"
        use_input_wav=true
        ;;
    *)
        echo "No such models: ${models}"
        exit 1
        ;;
esac

function download_models () {
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
if [ -z "${txt}" ]; then
    download_models
    txt=$(find ${download_dir}/${models}/etc -name "*.txt" | head -n 1)
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
if [ ! -f "${input_wav}" ] && [ ${use_input_wav} ]; then
    echo "No such WAV file for extracting meta information: ${input_wav}"
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
if [ ! -f "${txt}" ]; then
    echo "No such txt file: ${txt}"
    exit 1
fi

base=$(basename $txt .txt)
decode_dir=${decode_dir}/${base}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    mkdir -p ${decode_dir}/data
    echo "$base ${decode_dir}/data/dummy.wav" > ${decode_dir}/data/wav.scp
    echo "X $base" > ${decode_dir}/data/spk2utt
    echo "$base X" > ${decode_dir}/data/utt2spk
    echo -n "$base " > ${decode_dir}/data/text
    cat $txt >> ${decode_dir}/data/text
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Dummy Feature Generation"

    sox -n -r ${fs} -c 1 -b 16 ${decode_dir}/data/dummy.wav trim 0.000 0.001
    make_fbank.sh --cmd "${train_cmd}" --nj 1 \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        ${decode_dir}/data \
        ${decode_dir}/log \
        ${decode_dir}/fbank

    feat_synth_dir=${decode_dir}/dump; mkdir -p ${feat_synth_dir}
    dump.sh --cmd "$train_cmd" --nj 1 --do_delta false \
        ${decode_dir}/data/feats.scp ${cmvn} ${decode_dir}/log ${feat_synth_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Json Data Preparation"

    feat_synth_dir=${decode_dir}/dump
    data2json.sh --feat ${feat_synth_dir}/feats.scp \
        ${decode_dir}/data ${dict} > ${feat_synth_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${use_input_wav} ]; then
    echo "stage 3: x-vector extraction"

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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"

    feat_synth_dir=${decode_dir}/dump
    ${decode_cmd} ${decode_dir}/log/decode.log \
        tts_decode.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --debugmode ${debugmode} \
        --verbose ${verbose} \
        --out ${decode_dir}/outputs/feats \
        --json ${feat_synth_dir}/data.json \
        --model ${synth_model}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis"

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
