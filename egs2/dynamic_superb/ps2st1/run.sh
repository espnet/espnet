#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
ngpu=1

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh

# Stage 1-5: Data preparation
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 1-5: Data preparation"
    ./local/data.sh
fi

# Stage 12: Inference using integrated Qwen2-Audio model
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "Stage 12: Qwen2-Audio Inference with ESPnet2 integration"

    for dset in ./data/*; do
        _dset=${dset}
        echo ${_dset}
        _dir="exp/qwen2audio_inference/$(basename ${_dset})"
        _logdir="${_dir}/log"

        mkdir -p "${_logdir}"

        ${cuda_cmd} --gpu "${ngpu}" "${_logdir}/inference.log" \
            python -m espnet2.bin.ps2st_inference \
                --ngpu "${ngpu}" \
                --data_path_and_name_and_type "${_dset}/wav.scp,speech,sound" \
                --data_path_and_name_and_type "${_dset}/text.input,text,text" \
                --output_dir "${_dir}" \
                --batch_size 1 \
                --decode_config_path "conf/decode_asr.yaml"
    done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  echo "Stage 13: Scoring with score_sclite.sh"
  for dset in ./data/*; do
    echo ${dset}
    save_dir="exp/qwen2audio_inference/$(basename ${dset})"
    ref=${dset}/text.output
    hyp=${save_dir}/text

    mkdir -p ${save_dir}/score
    ./local/score.sh ${ref} ${hyp} ${save_dir}/score
  done
fi
