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

stage=1 #TODO(yiwen) try using versa, but some scores like vuv and semitone have not been support in VERSA
stop_stage=100
nj=8
inference_nj=8
gpu_inference=true
nbest=1

gen_dir=
ref_dir=
key_file=
eval_mcd=true
eval_log_F0=true
eval_semitone_ACC=true
eval_VUV_res=true
spk_config=conf/eval_spk.yaml
mcd_f0_config=conf/eval_mcd_f0.yaml

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
    log "Using espnet embeded eval functions"
    _dir="${gen_dir}/scoring"

    if ${eval_mcd}; then
        mkdir -p "${_dir}/MCD_res"
        # Objective Evaluation - MCD
        log "Begin Scoring for MCD metrics, results are written under ${_dir}/MCD_res"
        ${python} ./pyscripts/utils/evaluate_mcd.py \
            ${gen_dir}/wav \
            ${gen_dir}/ref_wav.scp \
            --outdir "${_dir}/MCD_res"
    fi

    if ${eval_log_F0}; then
        mkdir -p "${_dir}/F0_res"
        # Objective Evaluation - log-F0 RMSE
        log "Begin Scoring for F0 related metrics, results are written under ${_dir}/F0_res"
        ${python} pyscripts/utils/evaluate_f0.py \
            ${gen_dir}/wav \
            ${gen_dir}/ref_wav.scp \
            --outdir "${_dir}/F0_res"
    fi

    if ${eval_semitone_ACC}; then
        mkdir -p "${_dir}/SEMITONE_res"
        # Objective Evaluation - semitone ACC
        log "Begin Scoring for SEMITONE related metrics, results are written under ${_dir}/SEMITONE_res"
        ${python} pyscripts/utils/evaluate_semitone.py \
            ${gen_dir}/wav \
            ${gen_dir}/ref_wav.scp \
            --outdir "${_dir}/SEMITONE_res"
    fi

    if ${eval_VUV_res}; then
        mkdir -p "${_dir}/VUV_res"
        # Objective Evaluation - VUV error
        log "Begin Scoring for VUV related metrics, results are written under ${_dir}/VUV_res"
        ${python} pyscripts/utils/evaluate_vuv.py \
            ${gen_dir}/wav \
            ${gen_dir}/ref_wav.scp \
            --outdir "${_dir}/VUV_res"
    fi

fi

# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#     echo Use VERSA
#     for eval_item in mcd_f0; do
#         eval_flag=eval_${eval_item}
#         if ${!eval_flag}; then
#             # (1) init
#             opts=
#             eval_dir=${gen_dir}/scoring/eval_${eval_item}; mkdir -p ${eval_dir}

#             # (2) define pred, ref and config
#             if [ ${eval_item} == "mcd_f0" ]; then
#                 pred_file=${gen_dir}/wav.scp
#                 score_config=${mcd_f0_config}
#                 gt_file=${ref_dir}/wav.scp
#             elif [ ${eval_item} == "spk" ]; then
#                 pred_file=${gen_dir}/wav.scp
#                 score_config=${spk_config}
#                 gt_file=${ref_dir}/utt2spk
#             fi

#             # (3) split
#             _nj=$(min "${inference_nj}" "$(<${pred_file} wc -l)" )

#             split_files=""
#             for n in `seq ${_nj}`; do
#                 split_files+="${eval_dir}/pred.${n} "
#             done
#             utils/split_scp.pl ${pred_file} ${split_files}

#             if [ -n "${gt_file}" ]; then
#                 split_files=""
#                 for n in `seq ${_nj}`; do
#                     split_files+="${eval_dir}/gt.${n} "
#                 done
#                 utils/split_scp.pl ${gt_file} ${split_files}
#                 opts+="--gt ${eval_dir}/gt.JOB"
#             fi

#             (4) score
#             if ${gpu_inference}; then
#                 _cmd="${cuda_cmd}"
#                 _ngpu=1
#             else
#                 _cmd="${decode_cmd}"
#                 _ngpu=0
#             fi

#             ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${eval_dir}"/eval_${eval_item}.JOB.log \
#                 ${python} -m versa.bin.espnet_scorer \
#                     --pred ${eval_dir}/pred.JOB \
#                     --score_config ${score_config} \
#                     --use_gpu ${gpu_inference} \
#                     --output_file ${eval_dir}/result.JOB.txt \
#                     --io soundfile \
#                     ${opts} || exit 1;

#             # (5) aggregate
#             pyscripts/utils/aggregate_eval.py \
#                 --logdir ${eval_dir} \
#                 --scoredir ${eval_dir} \
#                 --nj ${_nj}

#         else
#             log "Skip evaluting ${eval_item}"
#         fi
#     done
# fi

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     log "Summarize the results"

#     all_eval_results=
#     metrics=

#     if ${eval_mcd}; then
#         all_eval_results+="${gen_dir}/scoring/MCD_res/mcd_avg_result.txt "
#         metrics+="mcd "
#     fi

#     if ${eval_log_F0}; then
#         all_eval_results+="${gen_dir}/scoring/F0_res/log_f0_rmse_avg_result.txt "
#         metrics+="log_F0 "
#     fi

#     if ${eval_semitone_ACC}; then
#         all_eval_results+="${gen_dir}/scoring/SEMITONE_res/semitone_acc_avg_result.txt "
#         metrics+="semitone_ACC "
#     fi

#     if ${eval_VUV_res}; then
#         all_eval_results+="${gen_dir}/scoring/VUV_res/vuv_error_avg_result.txt "
#         metrics+="VUV "
#     fi

#     ${python} pyscripts/utils/result_summary.py \
#         --all_eval_results ${all_eval_results} \
#         --key_file ${key_file} \
#         --output_dir ${gen_dir}/scoring \
#         --metrics ${metrics} \
#         --nbest ${nbest} \
#         > ${gen_dir}/scoring/final_result.txt
# fi

log "Successfully finished. [elapsed=${SECONDS}s]"
