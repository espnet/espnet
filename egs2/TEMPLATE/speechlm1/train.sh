#!/usr/bin/env bash

# Launch training from prepared SpeechLM datasets and length statistics.
set -euo pipefail

python=python
ngpu=1
num_nodes=1
node_rank=0
master_addr=
master_port=29500
train_config=
output_dir=exp/train
resume_path=
stats_dir=
train_unregistered_specifier=()
valid_unregistered_specifier=()
train_registered_specifier=()
valid_registered_specifier=()
save_loader_state=true
wandb_mode=disabled
wandb_project=speechlm
wandb_name=

help_message="Usage: $0 [options]

Train with espnet2.speechlm.bin.train using the active Python environment.
All relative paths are resolved from the recipe directory.

  --train-config PATH                 Training YAML
  --output-dir PATH                   Checkpoints and logs (default: exp/train)
  --stats-dir PATH                    Prepared stats_<task>_<name>.jsonl files
  --train-unregistered-specifier SPEC 'task:name:dataset.json[:factor]' (repeatable)
  --valid-unregistered-specifier SPEC 'task:name:dataset.json[:factor]' (repeatable)
  --train-registered-specifier SPEC   'task:name[:factor]' (repeatable)
  --valid-registered-specifier SPEC   'task:name[:factor]' (repeatable)
  --resume-path PATH                  Native .pt weights or DCP directory
                                      Omit to resume the latest output checkpoint
  --ngpu N                           GPUs per node (default: 1)
  --num-nodes N                      Number of nodes (default: 1)
  --node-rank N                      This node's rank (default: 0)
  --master-addr HOST                 Rank-0 host; required for multiple nodes
  --master-port PORT                 Rendezvous port (default: 29500)
  --save-loader-state true|false     Save batch assignments (default: true)
  --wandb-mode MODE                  disabled, offline, or online (default: disabled)
  --wandb-project NAME               Project name (default: speechlm)
  --wandb-name NAME                  Run name (default: derived from output-dir)
  --python PATH                     Python executable (default: python)
"

# Recipe run.sh enters its directory before calling this shared launcher.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
. "${repo_root}/egs2/TEMPLATE/speechlm1/parse_options.sh"

die() {
    echo "$0: $*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "Unexpected positional arguments: $*"
[[ -f ${train_config} ]] || die "Provide --train-config with an existing YAML file."
[[ -d ${stats_dir} ]] || die "Provide --stats-dir with prepared length statistics."
(( ${#train_unregistered_specifier[@]} + ${#train_registered_specifier[@]} > 0 )) \
    || die "Provide a training data specifier."
(( ${#valid_unregistered_specifier[@]} + ${#valid_registered_specifier[@]} > 0 )) \
    || die "Provide a validation data specifier."
[[ ${ngpu} =~ ^[1-9][0-9]*$ ]] || die "--ngpu must be a positive integer."
[[ ${num_nodes} =~ ^[1-9][0-9]*$ ]] || die "--num-nodes must be a positive integer."
[[ ${node_rank} =~ ^(0|[1-9][0-9]*)$ ]] || die "--node-rank must be a nonnegative integer."
(( node_rank < num_nodes )) || die "--node-rank must be less than --num-nodes."

if [[ -n ${resume_path} ]]; then
    [[ -f ${resume_path} || ( -d ${resume_path} && -f ${resume_path}/.metadata ) ]] \
        || die "--resume-path must be a weight file or DCP directory containing .metadata: ${resume_path}"
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
# Dataset resampling uses Python hashes; keep them consistent across ranks.
export PYTHONHASHSEED=0

launcher=("${python}" -m torch.distributed.run --nproc_per_node "${ngpu}")
if (( num_nodes == 1 )); then
    launcher+=(--standalone)
else
    [[ -n ${master_addr} ]] || die "Provide --master-addr for multiple nodes."
    launcher+=(--nnodes "${num_nodes}" --node_rank "${node_rank}"
        --master_addr "${master_addr}" --master_port "${master_port}")
fi

train_args=(
    --train-config "${train_config}"
    --output-dir "${output_dir}"
    --stats-dir "${stats_dir}"
    --wandb-mode "${wandb_mode}"
    --wandb-project "${wandb_project}"
)
for specifier in train_unregistered_specifier valid_unregistered_specifier \
    train_registered_specifier valid_registered_specifier; do
    specifier_ref=${specifier}[@]
    for spec in "${!specifier_ref}"; do
        train_args+=("--${specifier//_/-}" "${spec}")
    done
done
if [[ -n ${resume_path} ]]; then
    train_args+=(--resume-path "${resume_path}")
fi
if [[ ${save_loader_state} == true ]]; then
    train_args+=(--save-loader-state)
fi
if [[ -n ${wandb_name} ]]; then
    train_args+=(--wandb-name "${wandb_name}")
fi

exec "${launcher[@]}" --module espnet2.speechlm.bin.train "${train_args[@]}"
