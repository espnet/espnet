#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
stage=1
stop_stage=1
input_dir=
nj=1
ngpu_per_nj=1 
python=python3


log "$0 $*"
. utils/parse_options.sh

. ./cmd.sh

# NOTE(Jinchuan): make sure vLLM is installed before running this script

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p ${input_dir}/punctuate
    
    key_file=${input_dir}/text
    _nj=$(min "${nj}" "$(<${key_file} wc -l)")

    mkdir -p ${input_dir}/punctuate/logs
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${input_dir}/punctuate/logs/text.${n}"
    done
    utils/split_scp.pl ${key_file} ${split_scps}

    ${cuda_cmd} --gpu ${ngpu_per_nj} JOB=1:${_nj} ${input_dir}/punctuate/logs/punctuate.JOB.log \
        ${python} pyscripts/utils/llm_punctuate.py \
          -i ${input_dir}/punctuate/logs/text.JOB \
          -o ${input_dir}/punctuate/logs/text.punctuate.JOB \
          || exit 1;

    ${decode_cmd} JOB=1:${_nj} ${input_dir}/punctuate/logs/post_processing.JOB.log \
        ${python} pyscripts/text/llm_punctuate_postprocessing.py \
          -i ${input_dir}/punctuate/logs/text.JOB \
          -l ${input_dir}/punctuate/logs/text.punctuate.JOB \
          -o ${input_dir}/punctuate/logs/text.punctuate.JOB.replaced\
          || exit 1;
    
    (for n in `seq ${nj}`; do
        cat ${input_dir}/punctuate/logs/text.punctuate.${n}.replaced
    done) | sort >  ${input_dir}/text.punctuate
fi

log "Done! with ${SECONDS} seconds"






