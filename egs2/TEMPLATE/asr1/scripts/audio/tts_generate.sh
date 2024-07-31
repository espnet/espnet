#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

. ./path.sh
. ./cmd.sh

stage=1
stop_stage=100
nj=8
inference_nj=8
gpu_inference=false
n_examples=1
evaluation=false

src=
tgt=
tts_choice=chat_tts

python=python3

log "$0 $*"
. utils/parse_options.sh

for file in  text utt2spk; do
    if [ ! -f ${src}/${file} ]; then
        echo "cannot find ${src}/${file}. EXIT!" && exit 1; 
    fi
done

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "TTS generation by ${tts_choice}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    mkdir -p ${tgt}
    mkdir -p ${tgt}/logs

    _scp=${src}/text
    _nj=$(min "${inference_nj}" "$(<${_scp} wc -l)" )
    split_files=""
    for n in `seq ${_nj}`; do
        split_files+="${tgt}/logs/text.${n} "
    done
    utils/split_scp.pl ${_scp} ${split_files}

    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${tgt}"/logs/tts_generate.JOB.log \
        ${python} pyscripts/audio/tts_generate.py \
            --text ${tgt}/logs/text.JOB \
            --speaker_prompt ${src}/utt2spk \
            --n_examples ${n_examples} \
            --sample_rate 16000 \
            --tts_choice ${tts_choice} \
            --output_dir ${tgt}/logs/output.JOB || exit 1;

    for n in `seq ${_nj}`; do
        cat ${tgt}/logs/output.${n}/wav.scp
    done > ${tgt}/wav.scp

    awk -v N=${n_examples} '{{name=$1}for(i=0; i<N; i++){$1="tts_" name "_sample" i; print $0}}' \
        ${src}/text > ${tgt}/text
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if ${evaluation}; then
        mkdir -p ${tgt}/scoring
        cp ${tgt}/text ${tgt}/scoring/text
        ./scripts/utils/speechlm_eval/eval_tts.sh \
            --eval_spk false \
            --gen_dir ${tgt} \
            --ref_dir ${tgt}/scoring \
            --key_file ${tgt}/scoring/text \
            --nj ${nj} \
            --inference_nj ${inference_nj} \
            --gpu_inference ${gpu_inference} \
            --nbest ${n_examples}
    else
        echo "Skip evaluation"
    fi
fi