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
output_dir=exp/sft
stats_dir=
train_specifier=
valid_specifier=
train_unregistered_specifier=
valid_unregistered_specifier=
train_registered_specifier=
valid_registered_specifier=

# Export (stage 2)
export_dtype=bfloat16
export_path=
checkpoint_dir=

# Inference (stage 3)
train_config=
inference_config=
test_specifier=
test_unregistered_specifier=
test_registered_specifier=
inference_output_dir=
num_workers=1

help_message="Usage: $0 [--stage N] [--stop-stage N] [options]

  1  SFT        conf/train.yaml                -> exp/sft
  2  export     the latest complete checkpoint -> exp/sft/export/model.pt
  3  inference  those weights on --test-specifier

Stage 1 starts from --resume-path. Omit that option to continue exp/sft once
it has checkpoints of its own. Stage 3 can run on its own against published
weights: pass --export-path and --train-config.

  --stats-dir DIR                       Prepared length statistics
  --train-unregistered-specifier SPEC   Training data
  --valid-unregistered-specifier SPEC   Validation data
  --train-registered-specifier SPEC     Training data from the registry
  --valid-registered-specifier SPEC     Validation data from the registry
  Training can combine registered and unregistered data for each split.
  --train-config YAML                  Training or inference configuration
  --output-dir DIR                     Training output (default: exp/sft)
  --resume-path PATH                   Initialize weights, with fresh optimizer
  --checkpoint-dir DIR                 Select a DCP to export
  --export-path FILE                   Export destination or inference weights
  --inference-config YAML              Published decoding configuration
  --test-unregistered-specifier SPEC    Inference requests
  --test-registered-specifier SPEC      Inference requests from the registry
  Inference accepts exactly one of the two test specifier options.
  --inference-output-dir DIR           Decoding output (default: exp/sft/inference)
  --ngpu N --num-nodes N --node-rank N --master-addr HOST --master-port PORT
  --python PATH                        Active environment's Python
  --train-args 'OPTIONS'                Extra whitespace-separated train.sh options
"

repo_root=$(cd ../../.. && pwd)
. ../../TEMPLATE/asr1/utils/parse_options.sh

[[ $# -eq 0 ]] || die "Unexpected positional arguments: $*"
validate_stages "${stage}" "${stop_stage}" 3 "${node_rank}"
train_specifier=${train_unregistered_specifier:-${train_specifier}}
valid_specifier=${valid_unregistered_specifier:-${valid_specifier}}
test_specifier=${test_unregistered_specifier:-${test_specifier}}
export_path=${export_path:-${output_dir}/export/model.pt}
inference_output_dir=${inference_output_dir:-${output_dir}/inference}
read -r -a extra_train_args <<< "${train_args}"

if (( stage <= 1 && stop_stage >= 1 )); then
    log "Stage 1: SFT"
    [[ -n ${stats_dir} ]] || die "set --stats-dir"
    [[ -n ${train_specifier} || -n ${train_registered_specifier} ]] \
        || die "set --train-unregistered-specifier or --train-registered-specifier"
    [[ -n ${valid_specifier} || -n ${valid_registered_specifier} ]] \
        || die "set --valid-unregistered-specifier or --valid-registered-specifier"

    resume=$(resume_checkpoint "${output_dir}" "" "${resume_path}")
    resume_args=()
    if [[ -n ${resume} ]]; then
        resume_args=(--resume-path "${resume}")
    elif ! latest_dcp "${output_dir}/checkpoints" >/dev/null 2>&1; then
        die "set --resume-path, the published Bagpiper-Base base.pt, to start"
    fi

    ../../TEMPLATE/speechlm1/train.sh \
        --train-config "${train_config:-conf/train.yaml}" \
        --output-dir "${output_dir}" \
        --stats-dir "${stats_dir}" \
        --train-unregistered-specifier "${train_specifier}" \
        --valid-unregistered-specifier "${valid_specifier}" \
        --train-registered-specifier "${train_registered_specifier}" \
        --valid-registered-specifier "${valid_registered_specifier}" \
        --ngpu "${ngpu}" \
        --num-nodes "${num_nodes}" \
        --node-rank "${node_rank}" \
        ${master_addr:+--master-addr "${master_addr}"} \
        --master-port "${master_port}" \
        --wandb-project bagpiper-tts \
        --python "${python}" \
        ${resume_args[@]+"${resume_args[@]}"} ${extra_train_args[@]+"${extra_train_args[@]}"}
fi

if (( stage <= 2 && stop_stage >= 2 && node_rank == 0 )); then
    log "Stage 2: export the weights for inference"
    if [[ -z ${checkpoint_dir} ]]; then
        checkpoint_dir=$(latest_dcp "${output_dir}/checkpoints") \
            || die "no complete checkpoint under ${output_dir}/checkpoints"
    fi
    [[ -f ${checkpoint_dir}/.metadata ]] || die "incomplete DCP: ${checkpoint_dir}"
    [[ ! -e ${export_path} ]] \
        || die "${export_path} already exists; choose a new --export-path or run stage 3 to decode it"
    log "exporting ${checkpoint_dir}"
    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python}" -m espnet2.speechlm.bin.export_checkpoint \
        --checkpoint-dir "${checkpoint_dir}" \
        --output "${export_path}" \
        --dtype "${export_dtype}"
fi

if (( stage <= 3 && stop_stage >= 3 && node_rank == 0 )); then
    log "Stage 3: inference"
    if [[ -z ${train_config} ]]; then
        train_config=${output_dir}/train.yaml
        [[ -f ${train_config} ]] || train_config=conf/train.yaml
    fi
    [[ -n ${inference_config} ]] \
        || die "set --inference-config; the model repository publishes inference.yaml"
    [[ -n ${test_specifier} || -n ${test_registered_specifier} ]] \
        || die "set --test-unregistered-specifier or --test-registered-specifier; this recipe prepares no test data"
    [[ -z ${test_specifier} || -z ${test_registered_specifier} ]] \
        || die "set only one of --test-unregistered-specifier and --test-registered-specifier"
    test_args=(--test-unregistered-specifier "${test_specifier}")
    if [[ -n ${test_registered_specifier} ]]; then
        test_args=(--test-registered-specifier "${test_registered_specifier}")
    fi
    [[ -f ${export_path} ]] \
        || die "no weights at ${export_path}; run stage 2, or pass --export-path"

    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python}" -m espnet2.speechlm.bin.inference \
        --train-config "${train_config}" \
        --inference-config "${inference_config}" \
        --model-checkpoint "${export_path}" \
        "${test_args[@]}" \
        --output-dir "${inference_output_dir}" \
        --num-workers "${num_workers}"
    log "results: ${inference_output_dir}"
fi
