#!/usr/bin/env bash

# Bagpiper: warmup, pretraining, SFT, export, inference.
#
# Running it without options runs the five stages in order. --stage and
# --stop-stage select part of that, so --stage 3 --stop-stage 3 trains SFT
# alone and --stage 5 decodes without training anything.
#
# The data is yours to supply: this recipe prepares none. Set the variables
# below, or pass them on the command line. See README.md.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
. ../../TEMPLATE/speechlm1/stage_utils.sh

stage=1
stop_stage=5

# Launch
ngpu=1
num_nodes=1
node_rank=0
master_addr=
master_port=29500
python=python
train_args=          # anything else for train.sh, quoted

# Data for warmup and pretraining (stages 1 and 2)
stats_dir=
train_specifier=
valid_specifier=

# Data for SFT (stage 3): prepared instruction dialogues
sft_stats_dir=
sft_train_specifier=
sft_valid_specifier=

# Export (stage 4)
export_dtype=bfloat16
export_path=exp/sft/export/model.pt

# Inference (stage 5)
inference_config=
test_specifier=
inference_output_dir=exp/sft/inference
num_workers=1

help_message="Usage: $0 [--stage N] [--stop-stage N] [options]

  1  warmup       conf/train.yaml                 -> exp/warmup
  2  pretraining  conf/tuning/train_pretrain.yaml -> exp/pretrain
  3  SFT          conf/tuning/train_sft.yaml      -> exp/sft
  4  export       the latest complete checkpoint  -> ${export_path}
  5  inference    those weights on --test-specifier

A training stage continues its own output directory if it has checkpoints,
and otherwise starts from the stage before it. Stage 5 can run on its own
against published weights: pass --export-path and --train-config.
"

repo_root=$(cd ../../.. && pwd)
. "${repo_root}/egs2/TEMPLATE/asr1/utils/parse_options.sh"

train_config=${train_config:-}   # stage 5 only, for weights trained elsewhere

launch() {
    # One training stage: its configuration, where it writes, and where it
    # starts from. Everything else is the same for all three.
    local config=$1 output=$2 previous=$3
    local resume
    resume=$(resume_argument "${output}" "${previous}")
    local stats=$4 train_spec=$5 valid_spec=$6

    [[ -n ${stats} ]] || die "set --stats-dir (or --sft-stats-dir) for this stage"
    [[ -n ${train_spec} && -n ${valid_spec} ]] \
        || die "set the training and validation specifiers for this stage"

    # shellcheck disable=SC2086  # resume and train_args are argument lists
    ../../TEMPLATE/speechlm1/train.sh \
        --train-config "${config}" \
        --output-dir "${output}" \
        --stats-dir "${stats}" \
        --train-unregistered-specifier "${train_spec}" \
        --valid-unregistered-specifier "${valid_spec}" \
        --ngpu "${ngpu}" \
        --num-nodes "${num_nodes}" \
        --node-rank "${node_rank}" \
        ${master_addr:+--master-addr "${master_addr}"} \
        --master-port "${master_port}" \
        --wandb-project bagpiper \
        --python "${python}" \
        ${resume} ${train_args}
}

if (( stage <= 1 && stop_stage >= 1 )); then
    log "Stage 1: warmup"
    launch conf/train.yaml exp/warmup "" \
        "${stats_dir}" "${train_specifier}" "${valid_specifier}"
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    log "Stage 2: pretraining"
    launch conf/tuning/train_pretrain.yaml exp/pretrain exp/warmup \
        "${stats_dir}" "${train_specifier}" "${valid_specifier}"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
    log "Stage 3: SFT"
    launch conf/tuning/train_sft.yaml exp/sft exp/pretrain \
        "${sft_stats_dir}" "${sft_train_specifier}" "${sft_valid_specifier}"
fi

if (( stage <= 4 && stop_stage >= 4 )); then
    log "Stage 4: export the weights for inference"
    if [[ -f ${export_path} ]]; then
        log "${export_path} exists already; keeping it"
    else
        checkpoint_dir=$(latest_dcp exp/sft/checkpoints) \
            || die "no complete checkpoint under exp/sft/checkpoints"
        log "exporting ${checkpoint_dir}"
        mkdir -p "$(dirname -- "${export_path}")"
        PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${python}" -m espnet2.speechlm.bin.export_checkpoint \
            --checkpoint-dir "${checkpoint_dir}" \
            --output "${export_path}" \
            --dtype "${export_dtype}"
    fi
fi

if (( stage <= 5 && stop_stage >= 5 )); then
    log "Stage 5: inference"
    train_config=${train_config:-conf/tuning/train_sft.yaml}
    [[ -n ${inference_config} ]] \
        || die "set --inference-config; the model repositories publish inference_audio.yaml and inference_text.yaml"
    [[ -n ${test_specifier} ]] \
        || die "set --test-specifier, as 'dialogue:test:/path/to/test.json'; this recipe prepares no test data"
    [[ -f ${export_path} ]] \
        || die "no weights at ${export_path}; run stage 4, or pass --export-path"

    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python}" -m espnet2.speechlm.bin.inference \
        --train-config "${train_config}" \
        --inference-config "${inference_config}" \
        --model-checkpoint "${export_path}" \
        --test-unregistered-specifier "${test_specifier}" \
        --output-dir "${inference_output_dir}" \
        --num-workers "${num_workers}"
    log "results: ${inference_output_dir}"
fi
