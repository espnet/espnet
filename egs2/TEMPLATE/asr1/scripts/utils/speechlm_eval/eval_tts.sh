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
gpu_inference=true
nbest=1

gen_dir=
ref_dir=
key_file=
eval_wer=true
eval_spk=true
eval_mos=true
spk_config=conf/eval_spk.yaml
mos_config=conf/eval_mos.yaml

# wer options
whisper_tag=large
whisper_dir=local/whisper
cleaner=whisper_en
hyp_cleaner=whisper_en

python=python3

log "$0 $*"
. utils/parse_options.sh

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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # User VERSA
    for eval_item in mos spk; do
        eval_flag=eval_${eval_item}
        if ${!eval_flag}; then
            # (1) init
            opts=
            eval_dir=${gen_dir}/scoring/eval_${eval_item}; mkdir -p ${eval_dir}

            # (2) define pred, ref and config
            if [ ${eval_item} == "mos" ]; then
                pred_file=${gen_dir}/wav.scp
                score_config=${mos_config}
                gt_file=
            elif [ ${eval_item} == "spk" ]; then
                pred_file=${gen_dir}/wav.scp
                score_config=${spk_config}
                gt_file=${ref_dir}/utt2spk
            fi
            
            # (3) split
            _nj=$(min "${inference_nj}" "$(<${pred_file} wc -l)" )

            split_files=""
            for n in `seq ${_nj}`; do
                split_files+="${eval_dir}/pred.${n} "
            done
            utils/split_scp.pl ${pred_file} ${split_files}

            if [ -n "${gt_file}" ]; then
                split_files=""
                for n in `seq ${_nj}`; do
                    split_files+="${eval_dir}/gt.${n} "
                done
                utils/split_scp.pl ${gt_file} ${split_files}
                opts+="--gt ${eval_dir}/gt.JOB"
            fi

            # (4) score
            if ${gpu_inference}; then
                _cmd="${cuda_cmd}"
                _ngpu=1
            else
                _cmd="${decode_cmd}"
                _ngpu=0
            fi

            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${eval_dir}"/eval_${eval_item}.JOB.log \
                ${python} -m speech_evaluation.bin.espnet_scorer \
                    --pred ${eval_dir}/pred.JOB \
                    --score_config ${score_config} \
                    --use_gpu ${gpu_inference} \
                    --output_file ${eval_dir}/result.JOB.txt \
                    --io soundfile \
                    ${opts} || exit 1;
            
            # (5) aggregate
            pyscripts/utils/aggregate_eval.py \
                --logdir ${eval_dir} \
                --scoredir ${eval_dir} \
                --nj ${_nj}

        else
            log "Skip evaluting ${eval_item}"
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Summarize the results"

    all_eval_results=
    metrics=

    if ${eval_wer}; then
        all_eval_results+="${gen_dir}/scoring/eval_wer/utt_result.txt "
        metrics+="wer "
    fi

    if ${eval_spk}; then
        all_eval_results+="${gen_dir}/scoring/eval_spk/utt_result.txt "
        metrics+="spk_similarity "
    fi

    if ${eval_spk}; then
        all_eval_results+="${gen_dir}/scoring/eval_mos/utt_result.txt "
        metrics+="utmos "
    fi

    ${python} pyscripts/utils/result_summary.py \
        --all_eval_results ${all_eval_results} \
        --key_file ${key_file} \
        --output_dir ${gen_dir}/scoring \
        --metrics ${metrics} \
        --nbest ${nbest} \
        > ${gen_dir}/scoring/final_result.txt
fi

log "Successfully finished. [elapsed=${SECONDS}s]"




