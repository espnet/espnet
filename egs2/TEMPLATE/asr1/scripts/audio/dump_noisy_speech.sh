#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

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

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
nj=4                # number of parallel jobs
python=python3      # Specify python to execute espnet commands.

input_scp=
noise_scp=none
rir_scp=none
output_dir=

preprocessor=common
fs=16000
noise_apply_prob=1.0
noise_db_range="-5_20"
rir_apply_prob=0.5

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1
. ./cmd.sh || exit 1

if [ $# -ne 0 ]; then
    echo "Usage: $0 --input_scp foo1.scp --noise_scp foo2.scp --output_dir <noise_data>"
    exit 0
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    
    mkdir -p ${output_dir}
    mkdir -p ${output_dir}/local
    mkdir -p ${output_dir}/data
    mkdir -p ${output_dir}/logs

    # (1) only select a ratio of the input scp data.
    
    nutt=$(<${input_scp} wc -l)
    nutt_noise=$(<${noise_scp} wc -l)
    _nj=$(min ${nj} ${nutt} ${nutt_noise})

    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${output_dir}/logs/input.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl ${input_scp} ${split_scps} || exit 1;

    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${output_dir}/logs/noise.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl ${noise_scp} ${split_scps} || exit 1;

    ${decode_cmd} JOB=1:${_nj} ${output_dir}/logs/dump_noisy_speech.JOB.log \
        ${python} pyscripts/audio/dump_noisy_speech.py \
            --input_scp=${output_dir}/logs/input.JOB.scp \
            --noise_scp=${output_dir}/logs/noise.JOB.scp \
            --noise_apply_prob=${noise_apply_prob} \
            --noise_db_range="${noise_db_range}" \
            --rir_scp=${rir_scp} \
            --rir_apply_prob=${rir_apply_prob} \
            --output_dir=${output_dir}/data/JOB \
            --preprocessor=${preprocessor} \
            --fs=${fs}
    
    for n in `seq ${_nj}`; do
        cat ${output_dir}/data/${n}/wav.scp
    done > ${output_dir}/wav.scp

    for n in `seq ${_nj}`; do
        cat ${output_dir}/logs/input.${n}.scp
    done > ${output_dir}/spk1.scp
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
