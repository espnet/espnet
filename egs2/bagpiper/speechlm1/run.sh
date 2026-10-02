#!/usr/bin/env bash

# Bagpiper, from a checkpoint to decoded output.
#
# Training runs in three stages, each starting from the one before it:
#
#   ./run.sh --train-stage warmup   --ngpu 8 --stats-dir ... --train-unregistered-specifier ...
#   ./run.sh --train-stage pretrain --ngpu 8 ...
#   ./run.sh --train-stage sft      --ngpu 8 ...
#
# then
#
#   ./run.sh --stage export
#   ./run.sh --stage infer --inference-config inference_audio.yaml \
#       --test-unregistered-specifier 'dialogue:test:/path/to/test.json'
#
# A stage starts from the latest complete checkpoint of the stage before it,
# and continues its own if it has one, so re-running an interrupted stage
# picks it up rather than starting it again. The data is yours to supply:
# see README.md. Training options are those of
# ../../TEMPLATE/speechlm1/train.sh and are passed straight through;
# ./run.sh --help lists the rest.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

train_stage=warmup
rest=()
while (( $# > 0 )); do
    case $1 in
        --train-stage) train_stage=$2; shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done

case ${train_stage} in
    warmup)
        train_config=conf/train.yaml
        output_dir=exp/warmup
        resume_from=
        ;;
    pretrain)
        train_config=conf/tuning/train_pretrain.yaml
        output_dir=exp/pretrain
        resume_from=exp/warmup
        ;;
    sft)
        train_config=conf/tuning/train_sft.yaml
        output_dir=exp/sft
        resume_from=exp/pretrain
        ;;
    *)
        echo "$0: unknown --train-stage '${train_stage}'; expected warmup, pretrain or sft" >&2
        exit 1
        ;;
esac

exec ../../TEMPLATE/speechlm1/run.sh \
    --train-config "${train_config}" \
    --output-dir "${output_dir}" \
    ${resume_from:+--resume-from "${resume_from}"} \
    --wandb-project bagpiper \
    "${rest[@]+"${rest[@]}"}"
