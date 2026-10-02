#!/usr/bin/env bash

# Train, export and decode, for the recipes built on train.sh.
#
# train.sh launches training and nothing else, which left export and
# inference as commands to copy out of a README. This runs them as stages,
# so a recipe's run.sh covers the whole path from a checkpoint to decoded
# output in one file.
set -euo pipefail

stage=train
stop_stage=
python=python

# Stage "train": every option is passed through to train.sh unchanged.
# These two are read here as well, because export and infer need them.
train_config=
output_dir=exp/train
resume_from=
resume_path_given=

# Stage "export": the DCP that training wrote, as a single .pt file.
checkpoint_dir=
export_path=
export_dtype=

# Stage "infer": that .pt file, on a test manifest.
inference_config=
test_unregistered_specifier=
test_registered_specifier=
inference_output_dir=
num_workers=1

help_message="Usage: $0 --stage train|export|infer [options]

Run a SpeechLM recipe from training through to decoded output. Stages run
in order from --stage to --stop-stage. Both default to training alone,
because the data to decode is not prepared here: a full pass needs a test
manifest you provide, and then

    ./run.sh --stage train --stop-stage infer --inference-config ... \
        --test-unregistered-specifier ...

runs the three in one go.

  --stage STAGE                 First stage to run: train, export or infer
                                (default: train)
  --stop-stage STAGE            Last stage to run (default: the same stage,
                                so --stage export runs only the export)

 train (see train.sh --help for the rest)
  --train-config PATH           Training YAML
  --output-dir PATH             Checkpoints and logs (default: exp/train)
  --resume-from DIR             Start this stage from the latest complete
                                checkpoint of DIR, another stage's output
                                directory. Ignored once --output-dir has a
                                checkpoint of its own, so re-running an
                                interrupted stage continues it rather than
                                starting it again. An explicit --resume-path
                                wins.
  (every other option goes to train.sh unchanged)

 export
  --checkpoint-dir PATH         DCP directory to export
                                (default: the latest complete step_* under
                                <output-dir>/checkpoints)
  --export-path PATH            Where to write the weights
                                (default: <output-dir>/export/model.pt)
  --export-dtype DTYPE          float32, bfloat16 or float16

 infer
  --inference-config PATH       Decoding YAML; the model repositories publish
                                inference_audio.yaml and inference_text.yaml
  --test-unregistered-specifier SPEC  'task:name:dataset.json[:factor] ...'
  --test-registered-specifier SPEC    'task:name[:factor] ...'
  --inference-output-dir PATH   Results (default: <output-dir>/inference)
  --num-workers N               Decoding processes (default: 1)

  --python PATH                 Python executable (default: python)
"

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${here}/../../.." && pwd)

die() {
    echo "$0: $*" >&2
    exit 1
}

# Options this script owns are consumed here; everything else is forwarded to
# train.sh untouched, so a command that worked when run.sh called train.sh
# directly still works. parse_options.sh cannot do that - it exits on an
# option it does not know, which is every train.sh option.
forward=()
while (( $# > 0 )); do
    case $1 in
        --help|-h) echo "${help_message}"; exit 0 ;;
        --stage) stage=$2; shift 2 ;;
        --stop-stage) stop_stage=$2; shift 2 ;;
        --checkpoint-dir) checkpoint_dir=$2; shift 2 ;;
        --export-path) export_path=$2; shift 2 ;;
        --export-dtype) export_dtype=$2; shift 2 ;;
        --inference-config) inference_config=$2; shift 2 ;;
        --test-unregistered-specifier) test_unregistered_specifier=$2; shift 2 ;;
        --test-registered-specifier) test_registered_specifier=$2; shift 2 ;;
        --inference-output-dir) inference_output_dir=$2; shift 2 ;;
        --num-workers) num_workers=$2; shift 2 ;;
        # read by the export and infer stages, and still train.sh's to act on
        --train-config) train_config=$2; forward+=("$1" "$2"); shift 2 ;;
        --output-dir) output_dir=$2; forward+=("$1" "$2"); shift 2 ;;
        --resume-from) resume_from=$2; shift 2 ;;
        --resume-path) resume_path_given=1; forward+=("$1" "$2"); shift 2 ;;
        --python) python=$2; forward+=("$1" "$2"); shift 2 ;;
        *) forward+=("$1"); shift ;;
    esac
done

stage_index() {
    case $1 in
        train) echo 1 ;;
        export) echo 2 ;;
        infer) echo 3 ;;
        *) die "unknown stage '$1'; expected train, export or infer" ;;
    esac
}

# Naming one stage runs that stage; run several by naming both ends.
stop_stage=${stop_stage:-${stage}}
first=$(stage_index "${stage}")
last=$(stage_index "${stop_stage}")
(( first <= last )) || die "--stage ${stage} comes after --stop-stage ${stop_stage}"

run_stage() {
    local index
    index=$(stage_index "$1")
    (( first <= index && index <= last ))
}

latest_dcp() {
    # The highest step_N that finished writing: a DCP without .metadata is a
    # checkpoint that was interrupted, and exporting it fails deep inside the
    # reader rather than here.
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

if run_stage train; then
    echo "=== stage train ==="
    [[ -n ${train_config} ]] || die "--train-config is required for the train stage"
    if [[ -n ${resume_from} && -z ${resume_path_given} ]]; then
        if latest_dcp "${output_dir}/checkpoints" >/dev/null 2>&1; then
            # train.sh picks the latest checkpoint up by itself, and restores
            # the optimizer and step with it; handing it --resume-path here
            # would start this stage over with a fresh optimizer.
            echo "${output_dir} already has checkpoints; continuing it"
        else
            previous=$(latest_dcp "${resume_from}/checkpoints") \
                || die "no complete checkpoint under ${resume_from}/checkpoints to start from"
            echo "starting from ${previous}"
            forward+=(--resume-path "${previous}")
        fi
    fi
    "${here}/train.sh" "${forward[@]}"
fi

if run_stage export; then
    echo "=== stage export ==="
    if [[ -z ${checkpoint_dir} ]]; then
        checkpoint_dir=$(latest_dcp "${output_dir}/checkpoints") \
            || die "no complete checkpoint under ${output_dir}/checkpoints; pass --checkpoint-dir"
        echo "exporting the latest complete checkpoint: ${checkpoint_dir}"
    fi
    [[ -f ${checkpoint_dir}/.metadata ]] \
        || die "${checkpoint_dir} is not a complete DCP directory (no .metadata)"
    export_path=${export_path:-${output_dir}/export/model.pt}
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

if run_stage infer; then
    echo "=== stage infer ==="
    [[ -n ${train_config} ]] || die "--train-config is required for the infer stage"
    [[ -n ${inference_config} ]] \
        || die "--inference-config is required for the infer stage; the model repositories publish inference_audio.yaml and inference_text.yaml"
    [[ -n ${test_unregistered_specifier} || -n ${test_registered_specifier} ]] \
        || die "provide --test-unregistered-specifier or --test-registered-specifier; this recipe prepares no test data of its own"
    export_path=${export_path:-${output_dir}/export/model.pt}
    [[ -f ${export_path} ]] \
        || die "no exported weights at ${export_path}; run the export stage first"
    inference_output_dir=${inference_output_dir:-${output_dir}/inference}

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
