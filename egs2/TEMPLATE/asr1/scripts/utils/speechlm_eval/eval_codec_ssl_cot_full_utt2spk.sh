#!/usr/bin/env bash

# Copyright 2024Siddhant Arora
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Combined evaluation script with 4 stages:
# Stage 1: Basic text scoring (CER/WER using sclite)
# Stage 2: ASR evaluation (using Whisper)
# Stage 3: Quality evaluation (MOS/Speaker similarity using VERSA)
# Stage 4: Audio Quality Results summarization
# Stage 5: Semantic Evaluation

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

# Stage control
stage=1
stop_stage=5

# Parallelization options
nj=8
inference_nj=8
gpu_inference=true
nbest=1

# Directory options
gen_dir=
ref_dir=
key_file=

# Evaluation flags
eval_basic_scoring=true  # Stage 1: Basic CER/WER scoring
eval_wer=true            # Stage 2: ASR-based WER evaluation
eval_spk=true            # Stage 3: Speaker similarity evaluation
eval_mos=true            # Stage 3: MOS evaluation

# Stage 1 options (basic scoring)
scoring_metrics="cer wer"
nlsyms_txt=none

# Stage 2 options (ASR evaluation)
whisper_tag=large
whisper_dir=local/whisper

# Stage 3 options (quality evaluation)
spk_config=conf/eval_spk.yaml
mos_config=conf/eval_mos.yaml

# Text cleaning options
cleaner=whisper_en
hyp_cleaner=whisper_en

python=python3
gt_response_file=dump/raw_codec_ssl_cot_full_utt2spk_librispeech_100/eval2000/index_files/text

log "$0 $*"
. utils/parse_options.sh

mkdir -p ${gen_dir}/scoring

# =============================================================================
# Stage 1: Basic Text Scoring (CER/WER using sclite)
# =============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if ${eval_basic_scoring}; then
        log "Stage 1: Basic Text Scoring (CER/WER)"
        
        for _type in ${scoring_metrics}; do
            _scoredir="${gen_dir}/scoring/score_${_type}"
            mkdir -p "${_scoredir}"

            gen_text=${gen_dir}/src_text
            gt_text=${ref_dir}/src_text

            if [ "${_type}" = wer ]; then
                paste \
                    <(<"${gt_text}" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type word \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${cleaner}" \
                              ) \
                    <(<"${gt_text}" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/ref.trn"
                paste \
                    <(<"${gen_text}"  \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type word \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${hyp_cleaner}" \
                              ) \
                    <(<"${gen_text}" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

            elif [ "${_type}" = cer ]; then
                paste \
                    <(<"${gt_text}" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${cleaner}" \
                              ) \
                    <(<"${gt_text}" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/ref.trn"
                paste \
                    <(<"${gen_text}" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${hyp_cleaner}" \
                              ) \
                    <(<"${gen_text}" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"
            fi

            # Scoring with sclite
            sclite \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    else
        log "Skip Stage 1: Basic Text Scoring"
    fi
fi

# =============================================================================
# Stage 2: ASR Evaluation (using Whisper)
# =============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if ${eval_wer}; then
        log "Stage 2: ASR Evaluation (using Whisper)"
        
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
            ${gen_dir}/tgt_wav.scp ${gen_dir}/scoring/eval_wer

        # Convert to result json file
        ./pyscripts/utils/speechlm_convert_asr_result.py \
            --ref_file ${gen_dir}/scoring/eval_wer/score_wer/ref.trn \
            --hyp_file ${gen_dir}/scoring/eval_wer/score_wer/hyp.trn \
            --out_file ${gen_dir}/scoring/eval_wer/utt_result.txt \
            --file_type trn
    else
        log "Skip Stage 2: ASR Evaluation (CER/WER/TER)"
    fi
fi

# =============================================================================
# Stage 3: Quality Evaluation (MOS and Speaker Similarity using VERSA)
# =============================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Quality Evaluation (MOS/Speaker Similarity)"
    
    # Evaluate both MOS and Speaker similarity
    for eval_item in mos spk; do
        eval_flag=eval_${eval_item}
        if ${!eval_flag}; then
            log "Evaluating ${eval_item}..."
            
            # (1) Initialize
            opts=
            eval_dir=${gen_dir}/scoring/eval_${eval_item}
            mkdir -p ${eval_dir}

            # (2) Define prediction, reference and config files
            if [ ${eval_item} == "mos" ]; then
                pred_file=${gen_dir}/tgt_wav.scp
                score_config=${mos_config}
                gt_file=
            elif [ ${eval_item} == "spk" ]; then
                pred_file=${gen_dir}/tgt_wav.scp
                score_config=${spk_config}
                gt_file=${ref_dir}/utt2spk
            fi

            # (3) Split files for parallel processing
            _nj=$(min "${inference_nj}" "$(<${pred_file} wc -l)")

            split_files=""
            for n in $(seq ${_nj}); do
                split_files+="${eval_dir}/pred.${n} "
            done
            utils/split_scp.pl ${pred_file} ${split_files}

            if [ -n "${gt_file}" ]; then
                split_files=""
                for n in $(seq ${_nj}); do
                    split_files+="${eval_dir}/gt.${n} "
                done
                utils/split_scp.pl ${gt_file} ${split_files}
                opts+="--gt ${eval_dir}/gt.JOB"
            fi

            # (4) Run scoring
            if ${gpu_inference}; then
                _cmd="${cuda_cmd}"
                _ngpu=1
            else
                _cmd="${decode_cmd}"
                _ngpu=0
            fi

            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${eval_dir}"/eval_${eval_item}.JOB.log \
                ${python} -m versa.bin.scorer \
                    --pred ${eval_dir}/pred.JOB \
                    --score_config ${score_config} \
                    --use_gpu ${gpu_inference} \
                    --output_file ${eval_dir}/result.JOB.txt \
                    --io soundfile \
                    ${opts} || exit 1;

            # (5) Aggregate results
            pyscripts/utils/aggregate_eval.py \
                --logdir ${eval_dir} \
                --scoredir ${eval_dir} \
                --nj ${_nj}

        else
            log "Skip evaluating ${eval_item}"
        fi
    done
fi

# =============================================================================
# Stage 4: Results Summarization
# =============================================================================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Summarize all evaluation results"

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

    if ${eval_mos}; then
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

    log "Final results written to: ${gen_dir}/scoring/final_result.txt"
    cat ${gen_dir}/scoring/final_result.txt
fi

# =============================================================================
# Stage 5: ROUGE, METEOR and Perplexity results
# =============================================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Semantic Evaluations results"

    python pyscripts/text/get_rouge_score_tts_best.py ${gt_response_file} ${gen_dir}/scoring/selected_examples ${gen_dir}/scoring/eval_wer/text
    python pyscripts/text/perplexity_speechLM_best.py ${gen_dir}/scoring/selected_examples ${gen_dir}/scoring/eval_wer/text
fi


log "Successfully finished all stages. [elapsed=${SECONDS}s]"