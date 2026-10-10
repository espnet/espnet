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
train_config=        # override a single training stage, or the inference config
output_dir=          # override the selected training stage's output directory
resume_path=         # initialize the first selected training stage explicitly
warmup_output_dir=exp/warmup
pretrain_output_dir=exp/pretrain
sft_output_dir=exp/sft

# Data for warmup and pretraining (stages 1 and 2)
stats_dir=
train_specifier=
valid_specifier=
train_unregistered_specifier=
valid_unregistered_specifier=
train_registered_specifier=
valid_registered_specifier=

# Data for SFT (stage 3): prepared instruction dialogues
sft_stats_dir=
sft_train_specifier=
sft_valid_specifier=
sft_train_unregistered_specifier=
sft_valid_unregistered_specifier=
sft_train_registered_specifier=
sft_valid_registered_specifier=

# Export (stage 4)
export_dtype=bfloat16
export_path=
checkpoint_dir=      # omit to select the latest complete SFT checkpoint

# Inference (stage 5)
inference_config=
test_specifier=
test_unregistered_specifier=
test_registered_specifier=
inference_output_dir=
num_workers=1

help_message="Usage: $0 [--stage N] [--stop-stage N] [options]

  1  warmup       conf/train.yaml                 -> exp/warmup
  2  pretraining  conf/tuning/train_pretrain.yaml -> exp/pretrain
  3  SFT          conf/tuning/train_sft.yaml      -> exp/sft
  4  export       the latest complete checkpoint  -> exp/sft/export/model.pt
  5  inference    those weights on --test-specifier

A training stage continues its own output directory if it has checkpoints,
and otherwise starts from the stage before it. Stage 5 can run on its own
against published weights: pass --export-path and --train-config.

  --stats-dir DIR                       Warmup/pretraining length statistics
  --train-unregistered-specifier SPEC   Warmup/pretraining data
  --valid-unregistered-specifier SPEC   Warmup/pretraining validation data
  --train-registered-specifier SPEC     Warmup/pretraining data from the registry
  --valid-registered-specifier SPEC     Validation data from the registry
  --sft-stats-dir DIR                   SFT length statistics
  --sft-train-unregistered-specifier SPEC  SFT training manifests
  --sft-valid-unregistered-specifier SPEC  SFT validation manifests
  --sft-train-registered-specifier SPEC  SFT training data from the registry
  --sft-valid-registered-specifier SPEC  SFT validation data from the registry
  --sft-train-specifier / --sft-valid-specifier are unregistered aliases.
  Stage 3 alone also accepts the ordinary --stats-dir and data options.
  Training can combine registered and unregistered data for each split.
  --train-config YAML                  Override one training stage or inference
  --output-dir DIR                     Override one training stage's directory
  --resume-path PATH                   Initialize the first selected stage
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
validate_stages "${stage}" "${stop_stage}" 5 "${node_rank}"
if [[ -n ${train_config} || -n ${output_dir} ]] && (( stage < 3 && stop_stage > stage )); then
    die "--train-config and --output-dir require selecting only one training stage"
fi
if [[ -n ${output_dir} ]]; then
    case ${stage} in
        1) warmup_output_dir=${output_dir} ;;
        2) pretrain_output_dir=${output_dir} ;;
        *) sft_output_dir=${output_dir} ;;
    esac
fi
train_specifier=${train_unregistered_specifier:-${train_specifier}}
valid_specifier=${valid_unregistered_specifier:-${valid_specifier}}
test_specifier=${test_unregistered_specifier:-${test_specifier}}
sft_train_specifier=${sft_train_unregistered_specifier:-${sft_train_specifier}}
sft_valid_specifier=${sft_valid_unregistered_specifier:-${sft_valid_specifier}}
if (( stage == 3 )); then
    sft_stats_dir=${sft_stats_dir:-${stats_dir}}
    # Fall back per split, keeping an explicit SFT selection separate from
    # the ordinary data options used by warmup/pretraining.
    if [[ -z ${sft_train_specifier} && -z ${sft_train_registered_specifier} ]]; then
        sft_train_specifier=${train_specifier}
        sft_train_registered_specifier=${train_registered_specifier}
    fi
    if [[ -z ${sft_valid_specifier} && -z ${sft_valid_registered_specifier} ]]; then
        sft_valid_specifier=${valid_specifier}
        sft_valid_registered_specifier=${valid_registered_specifier}
    fi
fi
export_path=${export_path:-${sft_output_dir}/export/model.pt}
inference_output_dir=${inference_output_dir:-${sft_output_dir}/inference}
read -r -a extra_train_args <<< "${train_args}"

launch() {
    # One training stage: its configuration, where it writes, and where it
    # starts from. Everything else is the same for all three.
    local config=${train_config:-$1} output=$2 previous=$3
    local resume
    resume=$(resume_checkpoint "${output}" "${previous}" "${resume_path}")
    local resume_args=()
    if [[ -n ${resume} ]]; then
        resume_args=(--resume-path "${resume}")
    fi
    local stats=$4 train_spec=$5 valid_spec=$6
    local train_registered=$7 valid_registered=$8

    [[ -n ${stats} ]] || die "set --stats-dir (or --sft-stats-dir) for this stage"
    [[ -n ${train_spec} || -n ${train_registered} ]] \
        || die "set a registered or unregistered training specifier for this stage"
    [[ -n ${valid_spec} || -n ${valid_registered} ]] \
        || die "set a registered or unregistered validation specifier for this stage"

    ../../TEMPLATE/speechlm1/train.sh \
        --train-config "${config}" \
        --output-dir "${output}" \
        --stats-dir "${stats}" \
        --train-unregistered-specifier "${train_spec}" \
        --valid-unregistered-specifier "${valid_spec}" \
        --train-registered-specifier "${train_registered}" \
        --valid-registered-specifier "${valid_registered}" \
        --ngpu "${ngpu}" \
        --num-nodes "${num_nodes}" \
        --node-rank "${node_rank}" \
        ${master_addr:+--master-addr "${master_addr}"} \
        --master-port "${master_port}" \
        --wandb-project bagpiper \
        --python "${python}" \
        ${resume_args[@]+"${resume_args[@]}"} ${extra_train_args[@]+"${extra_train_args[@]}"}
    # An explicit initializer applies once; later stages chain normally.
    resume_path=
}

if (( stage <= 1 && stop_stage >= 1 )); then
    log "Stage 1: warmup"
    launch conf/train.yaml "${warmup_output_dir}" "" \
        "${stats_dir}" "${train_specifier}" "${valid_specifier}" \
        "${train_registered_specifier}" "${valid_registered_specifier}"
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    log "Stage 2: pretraining"
    launch conf/tuning/train_pretrain.yaml "${pretrain_output_dir}" "${warmup_output_dir}" \
        "${stats_dir}" "${train_specifier}" "${valid_specifier}" \
        "${train_registered_specifier}" "${valid_registered_specifier}"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
    log "Stage 3: SFT"
    launch conf/tuning/train_sft.yaml "${sft_output_dir}" "${pretrain_output_dir}" \
        "${sft_stats_dir}" "${sft_train_specifier}" "${sft_valid_specifier}" \
        "${sft_train_registered_specifier}" "${sft_valid_registered_specifier}"
fi

if (( stage <= 4 && stop_stage >= 4 && node_rank == 0 )); then
    log "Stage 4: export the weights for inference"
    if [[ -z ${checkpoint_dir} ]]; then
        checkpoint_dir=$(latest_dcp "${sft_output_dir}/checkpoints") \
            || die "no complete checkpoint under ${sft_output_dir}/checkpoints"
    fi
    [[ -f ${checkpoint_dir}/.metadata ]] || die "incomplete DCP: ${checkpoint_dir}"
    [[ ! -e ${export_path} ]] \
        || die "${export_path} already exists; choose a new --export-path or run stage 5 to decode it"
    log "exporting ${checkpoint_dir}"
    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python}" -m espnet2.speechlm.bin.export_checkpoint \
        --checkpoint-dir "${checkpoint_dir}" \
        --output "${export_path}" \
        --dtype "${export_dtype}"
fi

if (( stage <= 5 && stop_stage >= 5 && node_rank == 0 )); then
    log "Stage 5: inference"
    if [[ -z ${train_config} ]]; then
        train_config=${sft_output_dir}/train.yaml
        [[ -f ${train_config} ]] || train_config=conf/tuning/train_sft.yaml
    fi
    [[ -n ${inference_config} ]] \
        || die "set --inference-config; the model repositories publish inference_audio.yaml and inference_text.yaml"
    [[ -n ${test_specifier} || -n ${test_registered_specifier} ]] \
        || die "set --test-unregistered-specifier or --test-registered-specifier; this recipe prepares no test data"
    [[ -z ${test_specifier} || -z ${test_registered_specifier} ]] \
        || die "set only one of --test-unregistered-specifier and --test-registered-specifier"
    test_args=(--test-unregistered-specifier "${test_specifier}")
    if [[ -n ${test_registered_specifier} ]]; then
        test_args=(--test-registered-specifier "${test_registered_specifier}")
    fi
    [[ -f ${export_path} ]] \
        || die "no weights at ${export_path}; run stage 4, or pass --export-path"

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
