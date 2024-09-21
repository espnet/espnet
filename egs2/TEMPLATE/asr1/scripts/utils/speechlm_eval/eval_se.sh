#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Modified from eval_tts.sh
# Yoshiki Masuyama 09/21/2024

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
stop_stage=1
nj=8
inference_nj=8
gpu_inference=true
nbest=1
ngpu=1

gen_dir=
ref_dir=

eval_wer=true

# wer options
whisper_tag=large
whisper_dir=local/whisper
cleaner=whisper_en
hyp_cleaner=whisper_en

python=python3

log "$0 $*"
. utils/parse_options.sh

mkdir -p ${gen_dir}/scoring

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if ${eval_wer}; then
        # Use ESPnet builtin script
        ./scripts/utils/evaluate_asr.sh \
            --whisper_tag ${whisper_tag} \
            --whisper_dir ${whisper_dir} \
            --cleaner ${cleaner} \
            --hyp_cleaner ${hyp_cleaner} \
            --inference_nj ${inference_nj} \
            --nj ${nj} \
            --gt_text ${ref_dir}/text \
            --gpu_inference ${gpu_inference} \
            ${gen_dir}/wav.scp ${gen_dir}/scoring/eval_wer

        # convert to result json file
        ./pyscripts/utils/speechlm_convert_asr_result.py \
            --ref_file ${gen_dir}/scoring/eval_wer/score_wer/ref.trn \
            --hyp_file ${gen_dir}/scoring/eval_wer/score_wer/hyp.trn \
            --out_file ${gen_dir}/scoring/eval_wer/utt_result.txt \
            --file_type trn
    else
        log "Skip evaluating CER/WER/TER"
    fi
fi
