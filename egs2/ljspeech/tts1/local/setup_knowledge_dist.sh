#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#           2020 Songxiang Liu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Setup for knowledge distillation training in FastSpeech

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic setting
stage=0
stop_stage=100
ngpu=1
nj=32
verbose=1

# teacher model related
teacher_model_path=
teacher_model_config=

decode_config=conf/decode_for_knowledge_dist.yaml

# data related
tts_stats_dir=
feats_dir=dump/fbank
train_set=phn_train_nodev
dev_set=phn_dev


# filtering related
do_filtering=false
focus_rate_thres=0.65

outdir=

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check arguments
if [ -z "${teacher_model_path}" ]; then
    echo "you must set teacher_model_path." 2>&1
    exit 1;
fi
if [ -z "${teacher_model_config}" ]; then
  echo "you must set teacher_model_config." 2>&1
  exit 1;
fi

if [ ${ngpu} -ge 1 ]; then
  _cmd=${cuda_cmd}
else
  _cmd=${train_cmd}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Decoding for knowledge distillation."
  for dset in ${train_set} ${dev_set}; do
    _data=${feats_dir}/${dset}
    _outdir=${outdir}/${dset}
    _logdir="${_outdir}/log"
    mkdir -p ${_logdir}

    # 0. Copy feats_type
    cp "${_data}/feats_type" "${_outdir}/feats_type"

    # 1. Split the key file
    _scp=feats.scp
    _type=kaldi_ark
    key_file="${_data}/${_scp}"
    split_scps=""
    #_nj=$(min "${nj}" "$(<${key_file} wc -l)")
    _nj=${nj}
    for n in $(seq "${_nj}"); do
      split_scps+=" ${_outdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}
    
    # 2. Submit jobs
    ${_cmd} --gpu "${ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_decode.JOB.log \
      python3 -m espnet2.bin.tts_decode \
        --ngpu "${ngpu}" \
        --data_path_and_name_and_type "${_data}/text,text,text" \
        --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
        --key_file "${_outdir}"/keys.JOB.scp \
        --model_file "${teacher_model_path}" \
        --train_config "${teacher_model_config}" \
        --output_dir "${_outdir}"/feats.JOB \
        --config ${decode_config} \
        --allow_variable_data_keys true
    
    # 3. Concatenate the output files from each jobs
    for i in $(seq "${_nj}"); do
      cat "${_outdir}/feats.${i}.scp"
    done | LC_ALL=C sort -k1 > "${_outdir}/feats.scp"
    for i in $(seq "${_nj}"); do
      cat "${_outdir}/durations.${i}.scp"
    done | LC_ALL=C sort -k1 > "${_outdir}/durations.scp"
    for i in $(seq "${_nj}"); do
      cat "${_outdir}/focus_rates.${i}.scp"
    done | LC_ALL=C sort -k1 > "${_outdir}/focus_rates.scp"
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Update data for knowledge distillation training."
  for dset in ${train_set} ${dev_set}; do
    # perform filtering
    if [[ "${dset}" == *train_nodev* ]]; then
      speech_shape_scp="${tts_stats_dir}/train/speech_shape"
      text_shape_scp="${tts_stats_dir}/train/text_shape"
    else
      speech_shape_scp="${tts_stats_dir}/valid/speech_shape"
      text_shape_scp="${tts_stats_dir}/valid/text_shape"
    fi
    if ${do_filtering}; then
      local/filter_by_focus_rate.py \
        --focus-rates-scp "${outdir}/${dset}/focus_rates.scp" \
        --durations-scp "${outdir}/${dset}/durations.scp" \
        --speech-shape-scp "${speech_shape_scp}" \
        --text-shape-scp "${text_shape_scp}" \
        --threshold ${focus_rate_thres}
    fi
  done
  
  touch "${outdir}/.done"
  echo "successfully finished preparing data for knowledge distillation."
fi
    
