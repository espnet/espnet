#!/usr/bin/env bash

# Bagpiper-TTS: SFT, export, inference.
#
# Running it without options runs the three stages in order. --stage and
# --stop-stage select part of that, so --stage 3 decodes without training
# anything.
#
# There is one training stage, so it needs starting weights: the published
# Bagpiper-Base `base.pt`, as --resume-path. Repeat the command without it
# to continue an interrupted run - the latest complete checkpoint under
# exp/sft restores the model, optimizer and step.
#
# The data is yours to supply: this recipe prepares none. Set the variables
# below, or pass them on the command line. See README.md.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
. ../../TEMPLATE/speechlm1/stage_utils.sh

stage=1
stop_stage=3

# Launch
ngpu=1
num_nodes=1
node_rank=0
master_addr=
master_port=29500
python=python
train_args=          # anything else for train.sh, quoted

# Training (stage 1)
resume_path=         # /path/to/bagpiper-base/base.pt for a fresh run
stats_dir=
train_specifier=
valid_specifier=

# Export (stage 2)
export_dtype=bfloat16
export_path=exp/sft/export/model.pt

# Inference (stage 3)
train_config=
inference_config=
test_specifier=
inference_output_dir=exp/sft/inference
num_workers=1

help_message="Usage: $0 [--stage N] [--stop-stage N] [options]

  1  SFT        conf/train.yaml                -> exp/sft
  2  export     the latest complete checkpoint -> ${export_path}
  3  inference  those weights on --test-specifier

Stage 1 starts from --resume-path, and continues exp/sft instead once that
has checkpoints of its own. Stage 3 can run on its own against published
weights: pass --export-path and --train-config.
"

repo_root=$(cd ../../.. && pwd)
. "${repo_root}/egs2/TEMPLATE/asr1/utils/parse_options.sh"

if (( stage <= 1 && stop_stage >= 1 )); then
    log "Stage 1: SFT"
    [[ -n ${stats_dir} ]] || die "set --stats-dir"
    [[ -n ${train_specifier} && -n ${valid_specifier} ]] \
        || die "set --train-specifier and --valid-specifier"

    resume=
    if latest_dcp exp/sft/checkpoints >/dev/null 2>&1; then
        log "exp/sft already has checkpoints; continuing it"
    elif [[ -n ${resume_path} ]]; then
        log "starting from ${resume_path}"
        resume="--resume-path ${resume_path}"
    else
        die "set --resume-path, the published Bagpiper-Base base.pt, to start"
    fi

    # shellcheck disable=SC2086  # resume and train_args are argument lists
    ../../TEMPLATE/speechlm1/train.sh \
        --train-config conf/train.yaml \
        --output-dir exp/sft \
        --stats-dir "${stats_dir}" \
        --train-unregistered-specifier "${train_specifier}" \
        --valid-unregistered-specifier "${valid_specifier}" \
        --ngpu "${ngpu}" \
        --num-nodes "${num_nodes}" \
        --node-rank "${node_rank}" \
        ${master_addr:+--master-addr "${master_addr}"} \
        --master-port "${master_port}" \
        --wandb-project bagpiper-tts \
        --python "${python}" \
        ${resume} ${train_args}
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    log "Stage 2: export the weights for inference"
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

if (( stage <= 3 && stop_stage >= 3 )); then
    log "Stage 3: inference"
    train_config=${train_config:-conf/train.yaml}
    [[ -n ${inference_config} ]] \
        || die "set --inference-config; the model repository publishes inference.yaml"
    [[ -n ${test_specifier} ]] \
        || die "set --test-specifier, as 'dialogue:test:/path/to/test.json'; this recipe prepares no test data"
    [[ -f ${export_path} ]] \
        || die "no weights at ${export_path}; run stage 2, or pass --export-path"

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
