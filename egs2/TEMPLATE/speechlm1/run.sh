#!/usr/bin/env bash

# Train, export and decode, in order, for the recipes built on train.sh.
#
# train.sh launches one training run and nothing else, which left a
# recipe's schedule - which configuration, started from which checkpoint,
# then exported, then decoded - as commands to copy out of a README. A
# recipe declares its training stages here instead, and this runs them in
# order, followed by export and inference.
set -euo pipefail

# "name:config:output_dir name:config:output_dir ...", in the order they run.
train_stages=
stage=
stop_stage=
python=python

# Each training stage starts from the one before it. These override that.
resume_path=

# Stage "export": the DCP the last training stage wrote, as a single file.
checkpoint_dir=
export_path=
export_dtype=

# Stage "infer": that file, on a test manifest.
train_config=
inference_config=
test_unregistered_specifier=
test_registered_specifier=
inference_output_dir=
num_workers=1

help_message="Usage: $0 --train-stages 'name:config:dir ...' [options]

Run a recipe in order: its training stages, then export, then inference.
Stages run from --stage to --stop-stage.

  --stage NAME        First stage (default: the first training stage)
  --stop-stage NAME   Last stage. The default is export, because a decode
                      needs a test manifest these recipes do not prepare;
                      asking for a later stage carries the default along, so
                      --stage infer decodes

Training stages come from the recipe. Each starts from the latest complete
checkpoint of the stage before it, and continues its own output directory
once that has one, so repeating a command resumes an interrupted stage
rather than starting it again. An explicit --resume-path wins over both.
Every option this script does not recognise goes to train.sh unchanged.

 export
  --checkpoint-dir PATH   DCP to export (default: the latest complete one
                          of the last training stage)
  --export-path PATH      Where the weights go (default: <last stage>/export/model.pt)
  --export-dtype DTYPE    float32, bfloat16 or float16

 infer
  --train-config PATH     The model's configuration (default: the last
                          training stage's). Give it to decode with weights
                          this recipe did not train.
  --inference-config PATH Decoding YAML; the model repositories publish
                          inference_audio.yaml and inference_text.yaml
  --test-unregistered-specifier SPEC  'task:name:dataset.json[:factor] ...'
  --test-registered-specifier SPEC    'task:name[:factor] ...'
  --inference-output-dir PATH         Results (default: <last stage>/inference)
  --num-workers N                     Decoding processes (default: 1)

  --python PATH           Python executable (default: python)

Skipping training: --stage infer with --export-path and --train-config
decodes published weights without training anything.
"

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${here}/../../.." && pwd)

die() {
    echo "$0: $*" >&2
    exit 1
}

# Options this script owns are consumed here; everything else is forwarded to
# train.sh untouched. parse_options.sh cannot do that - it exits on an option
# it does not know, which is every train.sh option.
forward=()
while (( $# > 0 )); do
    case $1 in
        --help|-h) echo "${help_message}"; exit 0 ;;
        --train-stages) train_stages=$2; shift 2 ;;
        --stage) stage=$2; shift 2 ;;
        --stop-stage) stop_stage=$2; shift 2 ;;
        --resume-path) resume_path=$2; shift 2 ;;
        --checkpoint-dir) checkpoint_dir=$2; shift 2 ;;
        --export-path) export_path=$2; shift 2 ;;
        --export-dtype) export_dtype=$2; shift 2 ;;
        --train-config) train_config=$2; shift 2 ;;
        --inference-config) inference_config=$2; shift 2 ;;
        --test-unregistered-specifier) test_unregistered_specifier=$2; shift 2 ;;
        --test-registered-specifier) test_registered_specifier=$2; shift 2 ;;
        --inference-output-dir) inference_output_dir=$2; shift 2 ;;
        --num-workers) num_workers=$2; shift 2 ;;
        --python) python=$2; forward+=("$1" "$2"); shift 2 ;;
        *) forward+=("$1"); shift ;;
    esac
done

[[ -n ${train_stages} ]] || die "the recipe must pass --train-stages"

names=() configs=() outputs=()
for declaration in ${train_stages}; do
    IFS=: read -r name config output <<<"${declaration}"
    [[ -n ${name} && -n ${config} && -n ${output} ]] \
        || die "a training stage is 'name:config:output_dir', not '${declaration}'"
    names+=("${name}") configs+=("${config}") outputs+=("${output}")
done
# bash 3.2 has no negative index
last_output=${outputs[${#outputs[@]} - 1]}

order=("${names[@]}" export infer)
stage=${stage:-${order[0]}}

index_of() {
    local wanted=$1 position=1 name
    for name in "${order[@]}"; do
        if [[ ${name} == "${wanted}" ]]; then
            echo "${position}"
            return 0
        fi
        (( position++ ))
    done
    die "unknown stage '${wanted}'; this recipe has: ${order[*]}"
}

first=$(index_of "${stage}")
if [[ -n ${stop_stage} ]]; then
    last=$(index_of "${stop_stage}")
    (( first <= last )) \
        || die "--stage ${stage} comes after --stop-stage ${stop_stage}"
else
    # Stop after export by default: decoding needs a test manifest these
    # recipes do not prepare, so a bare run should not end in an error
    # about one. Asking for a later stage carries the default along.
    last=$(index_of export)
    (( last >= first )) || last=${first}
fi

latest_dcp() {
    # The highest step_N that finished writing: a DCP without .metadata is an
    # interrupted checkpoint, and exporting it fails deep inside the reader
    # rather than here.
    local directory="$1" candidate best=
    for candidate in "${directory}"/step_*; do
        [[ -f ${candidate}/.metadata ]] || continue
        if [[ -z ${best} || ${candidate##*step_} -gt ${best##*step_} ]]; then
            best=${candidate}
        fi
    done
    [[ -n ${best} ]] || return 1
    echo "${best}"
}

position=0
for name in "${names[@]}"; do
    (( position++ ))
    (( first <= position && position <= last )) || continue

    config=${configs[position - 1]}
    output=${outputs[position - 1]}
    echo "=== stage ${name} ==="

    stage_args=(--train-config "${config}" --output-dir "${output}")
    if [[ -n ${resume_path} ]]; then
        stage_args+=(--resume-path "${resume_path}")
    elif latest_dcp "${output}/checkpoints" >/dev/null 2>&1; then
        # train.sh finds the latest checkpoint itself and restores the
        # optimizer and step with it; handing it --resume-path here would
        # start the stage over with a fresh optimizer.
        echo "${output} already has checkpoints; continuing it"
    elif (( position > 1 )); then
        previous=${outputs[position - 2]}
        started_from=$(latest_dcp "${previous}/checkpoints") \
            || die "no complete checkpoint under ${previous}/checkpoints to start ${name} from"
        echo "starting from ${started_from}"
        stage_args+=(--resume-path "${started_from}")
    fi

    "${here}/train.sh" "${stage_args[@]}" "${forward[@]+"${forward[@]}"}"
done

export_path=${export_path:-${last_output}/export/model.pt}

if (( first <= $(index_of export) && $(index_of export) <= last )); then
    echo "=== stage export ==="
    if [[ -z ${checkpoint_dir} ]]; then
        checkpoint_dir=$(latest_dcp "${last_output}/checkpoints") \
            || die "no complete checkpoint under ${last_output}/checkpoints; pass --checkpoint-dir"
        echo "exporting the latest complete checkpoint: ${checkpoint_dir}"
    fi
    [[ -f ${checkpoint_dir}/.metadata ]] \
        || die "${checkpoint_dir} is not a complete DCP directory (no .metadata)"
    mkdir -p "$(dirname -- "${export_path}")"
    if [[ -f ${export_path} ]]; then
        echo "${export_path} exists already; keeping it"
    else
        export_args=(--checkpoint-dir "${checkpoint_dir}" --output "${export_path}")
        [[ -n ${export_dtype} ]] && export_args+=(--dtype "${export_dtype}")
        PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${python}" -m espnet2.speechlm.bin.export_checkpoint "${export_args[@]}"
    fi
fi

if (( first <= $(index_of infer) && $(index_of infer) <= last )); then
    echo "=== stage infer ==="
    train_config=${train_config:-${configs[${#configs[@]} - 1]}}
    [[ -f ${train_config} ]] \
        || die "no model configuration at ${train_config}; pass --train-config"
    [[ -n ${inference_config} ]] \
        || die "--inference-config is required to decode; the model repositories publish inference_audio.yaml and inference_text.yaml"
    [[ -n ${test_unregistered_specifier} || -n ${test_registered_specifier} ]] \
        || die "provide --test-unregistered-specifier or --test-registered-specifier; this recipe prepares no test data of its own"
    [[ -f ${export_path} ]] \
        || die "no weights at ${export_path}; run the export stage, or pass --export-path"
    inference_output_dir=${inference_output_dir:-${last_output}/inference}

    infer_args=(
        --train-config "${train_config}"
        --inference-config "${inference_config}"
        --model-checkpoint "${export_path}"
        --output-dir "${inference_output_dir}"
        --num-workers "${num_workers}"
    )
    [[ -n ${test_unregistered_specifier} ]] \
        && infer_args+=(--test-unregistered-specifier "${test_unregistered_specifier}")
    [[ -n ${test_registered_specifier} ]] \
        && infer_args+=(--test-registered-specifier "${test_registered_specifier}")
    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python}" -m espnet2.speechlm.bin.inference "${infer_args[@]}"
    echo "results: ${inference_output_dir}"
fi
