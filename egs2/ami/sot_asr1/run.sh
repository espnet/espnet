#!/usr/bin/env bash
# SOT multi-talker ASR recipe for AMI dataset using Whisper.
#
# Two modes:
#   1. Training (and stock-pipeline inference) — wraps asr.sh
#        ./run.sh --stage 11 --stop_stage 11        # train
#        ./run.sh --stage 1  --stop_stage 13        # full pipeline
#
#   2. Inference + evaluation against a checkpoint bundle
#      (model.pth + config.yaml + token_list.txt)
#        ./run.sh --inference_model exp/whisper-sot-small-ami \
#                 --whisper_model small
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/tuning/train_sot_asr_whisper_small.yaml
inference_config=conf/tuning/decode_sot.yaml

# Checkpoint-bundle inference defaults
inference_model=""
whisper_model="small"
decode_out="decode_inference"
decode_test_sets=""           # space-separated; defaults to ${test_sets}
fp16_flag="--fp16"            # fp16 on by default; pass --no-fp16 to disable
speaker_change_symbol=""      # optional; else decode.py reads it from config.yaml
do_score=true                 # score after decoding; pass --no_score to skip
score_cleaner="whisper_en"    # espnet TextCleaner used for cpWER
der_collar="0.25"             # md-eval.pl collar (seconds) for DER

# Pull our own flags out, forward the rest to asr.sh
asr_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --inference_model)       inference_model="$2";        shift 2 ;;
        --whisper_model)         whisper_model="$2";          shift 2 ;;
        --decode_out)            decode_out="$2";             shift 2 ;;
        --decode_test_sets)      decode_test_sets="$2";       shift 2 ;;
        --speaker_change_symbol) speaker_change_symbol="$2";  shift 2 ;;
        --score_cleaner)         score_cleaner="$2";          shift 2 ;;
        --der_collar)            der_collar="$2";             shift 2 ;;
        --fp16)                  fp16_flag="--fp16";          shift ;;
        --no-fp16)               fp16_flag="";                shift ;;
        --no_score)              do_score=false;              shift ;;
        *) asr_args+=("$1"); shift ;;
    esac
done

if [ -n "${inference_model}" ]; then
    # ----- Mode 2: checkpoint-bundle inference (+ scoring) -----
    if [ -z "${decode_test_sets}" ]; then
        decode_test_sets="${test_sets}"
    fi
    for tset in ${decode_test_sets}; do
        outdir="${inference_model}/${decode_out}/${tset}"
        echo "[run.sh] Decoding ${tset} -> ${outdir}"
        python local/decode.py "${inference_model}" \
            --whisper_model "${whisper_model}" \
            --wav_scp "data/${tset}/wav.scp" \
            --out_subdir "${decode_out}/${tset}" \
            ${speaker_change_symbol:+--speaker_change_symbol "${speaker_change_symbol}"} \
            ${fp16_flag}
        if "${do_score}"; then
            echo "[run.sh] Scoring ${tset} -> ${outdir}/scoring"
            local/score_sot.sh "${outdir}" "data/${tset}" \
                "${outdir}/scoring" "${score_cleaner}" "${der_collar}"
        fi
    done
    exit 0
fi

# ----- Mode 1: standard ESPnet training / stock inference via asr.sh -----
./asr.sh \
    --lang en \
    --feats_type raw \
    --token_type whisper_multilingual \
    --sot_asr false \
    --max_wav_duration 30 \
    --feats_normalize null \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "${asr_args[@]}"
