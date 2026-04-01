#!/usr/bin/env bash
# SOT multi-talker ASR recipe for AMI dataset.
#
# Uses native OpenAI Whisper encoder/decoder with tiktoken tokenization.
#
# Stage mapping (follows ESPnet asr.sh convention):
#   Stage 1:  Data preparation (Lhotse CutSet -> Kaldi format)
#   Stage 2:  Speed perturbation (skipped)
#   Stage 3:  Format wav.scp (skipped)
#   Stage 4:  Remove long/short data (skipped)
#   Stage 5:  Generate token list
#   Stage 6-9: LM training (skipped)
#   Stage 10: ASR collect stats (skipped)
#   Stage 11: ASR Training
#   Stage 12: Decoding
#   Stage 13: Scoring (cpWER)
#
# Usage:
#   # Full pipeline:
#   ./run.sh --stage 1 --use_timestamps true
#
#   # Train only:
#   ./run.sh --stage 11 --stop_stage 11
#
#   # Decode + evaluate:
#   ./run.sh --stage 12 --stop_stage 13

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# General
stage=1
stop_stage=13
ngpu=1
nj=8
expdir=exp
python=python3

# Data
train_set=train
valid_set=dev
test_sets="dev test"

# SOT data prep
use_timestamps=false
max_timestamp_pause=2.0
train_cutset=   # path to train cutset .jsonl.gz (required for stage 1)
valid_cutset=   # path to valid cutset .jsonl.gz (required for stage 1)
test_cutsets=   # space-separated paths to test cutset .jsonl.gz files

# Config
asr_config=conf/tuning/train_sot_small.yaml
decode_config=conf/tuning/decode_sot.yaml
token_list=  # path to token_list (auto-generated in stage 5 if empty)
added_tokens_file=local/added_tokens.txt

log "$0 $*"

# Inline parse_options (no dependency on utils/ symlink)
while true; do
    [ -z "${1:-}" ] && break
    case "$1" in
        --*) name=$(echo "$1" | sed 's/^--//' | sed 's/-/_/g')
             eval "${name}=\"$2\""
             shift 2 ;;
        *)   break ;;
    esac
done

# ================================
# Stage 1: Data preparation
# ================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Prepare SOT data from Lhotse CutSets"

    _sot_opts="--use_timestamps ${use_timestamps} \
        --max_timestamp_pause ${max_timestamp_pause} \
        --added_tokens_file ${added_tokens_file}"

    if [ -n "${train_cutset}" ]; then
        log "Preparing train set: ${train_set}"
        ${python} local/prepare_sot.py \
            --cutset_paths ${train_cutset} \
            --output_dir data/${train_set} \
            ${_sot_opts}
    fi

    if [ -n "${valid_cutset}" ]; then
        log "Preparing valid set: ${valid_set}"
        ${python} local/prepare_sot.py \
            --cutset_paths ${valid_cutset} \
            --output_dir data/${valid_set} \
            ${_sot_opts}
    fi

    if [ -n "${test_cutsets}" ]; then
        idx=0
        for dset in ${test_sets}; do
            cutset=$(echo ${test_cutsets} | cut -d' ' -f$((idx+1)))
            if [ -n "${cutset}" ]; then
                log "Preparing test set: ${dset}"
                ${python} local/prepare_sot.py \
                    --cutset_paths ${cutset} \
                    --output_dir data/${dset} \
                    ${_sot_opts}
            fi
            idx=$((idx+1))
        done
    fi

    # Validate data directories
    for dset in ${train_set} ${valid_set} ${test_sets}; do
        dir=data/${dset}
        if [ ! -d "${dir}" ]; then
            log "WARNING: ${dir} does not exist"
            continue
        fi
        for f in wav.scp text utt2spk; do
            if [ ! -f "${dir}/${f}" ]; then
                log "ERROR: Missing required file ${dir}/${f}"
                exit 1
            fi
        done
        log "Data directory ${dir}: $(wc -l < "${dir}/wav.scp") utterances"
    done
fi

# ================================
# Stage 2-4: Skipped (speed perturbation, format wav, remove long/short)
# ================================

# ================================
# Stage 5: Generate token list
# ================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ -z "${token_list}" ]; then
        token_list="${expdir}/token_list.txt"
    fi
    if [ ! -f "${token_list}" ]; then
        log "Stage 5: Generating token list from tiktoken"
        mkdir -p "${expdir}"
        ${python} local/generate_token_list.py \
            --output "${token_list}" \
            --added_tokens_txt "${added_tokens_file}"
    else
        log "Stage 5: Token list already exists: ${token_list}"
    fi
fi

# Set token_list default for later stages
if [ -z "${token_list}" ]; then
    token_list="${expdir}/token_list.txt"
fi

# ================================
# Stage 6-10: Skipped (LM training, ASR collect stats)
# ================================

# ================================
# Stage 11: ASR Training
# ================================
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: SOT ASR Training"

    _train_dir=data/${train_set}
    _valid_dir=data/${valid_set}

    _opts=""
    _opts+="--train_data_path_and_name_and_type ${_train_dir}/wav.scp,speech,sound "
    _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
    _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/wav.scp,speech,sound "
    _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "

    _tag=$(basename "${asr_config}" .yaml)
    _expdir="${expdir}/sot_${_tag}"

    ${python} -m espnet2.bin.sot_train \
        --config "${asr_config}" \
        --token_list "${token_list}" \
        --token_type whisper_multilingual \
        --output_dir "${_expdir}" \
        --ngpu ${ngpu} \
        --num_workers ${nj} \
        ${_opts} \
        "$@"
fi

# ================================
# Stage 12: Decoding
# ================================
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: SOT Decoding"

    _tag=$(basename "${asr_config}" .yaml)
    _expdir="${expdir}/sot_${_tag}"
    _model_file="${_expdir}/valid.loss.best.pth"

    if [ ! -f "${_model_file}" ]; then
        log "ERROR: Model file not found: ${_model_file}"
        exit 1
    fi

    for dset in ${test_sets}; do
        _data_dir=data/${dset}
        _decode_dir="${_expdir}/decode_${dset}"

        _decode_opts=""
        _decode_opts+="--data_path_and_name_and_type ${_data_dir}/wav.scp,speech,sound "

        ${python} -m espnet2.bin.sot_inference \
            --config "${decode_config}" \
            --asr_train_config "${_expdir}/config.yaml" \
            --asr_model_file "${_model_file}" \
            --output_dir "${_decode_dir}" \
            --ngpu ${ngpu} \
            ${_decode_opts} \
            "$@"

        log "Decoding results: ${_decode_dir}"
    done
fi

# ================================
# Stage 13: Scoring (cpWER)
# ================================
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "Stage 13: Evaluate cpWER"

    _tag=$(basename "${asr_config}" .yaml)
    _expdir="${expdir}/sot_${_tag}"

    for dset in ${test_sets}; do
        _decode_dir="${_expdir}/decode_${dset}"
        _hyp_text="${_decode_dir}/1best_recog/text"
        _ref_text="data/${dset}/text"
        _eval_dir="${_decode_dir}/eval"

        if [ ! -f "${_hyp_text}" ]; then
            log "WARNING: Hypothesis file not found: ${_hyp_text}, skipping ${dset}"
            continue
        fi

        ${python} local/evaluate_sot.py \
            --hyp_text "${_hyp_text}" \
            --ref_text "${_ref_text}" \
            --output_dir "${_eval_dir}"

        log "Evaluation results for ${dset}: ${_eval_dir}/cpwer.json"
        if [ -f "${_eval_dir}/cpwer.json" ]; then
            cat "${_eval_dir}/cpwer.json"
        fi
    done
fi

log "Done."
