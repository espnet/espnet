#!/usr/bin/env bash

# Bagpiper, in order: warmup, pretraining, SFT, export, inference.
#
#   ./run.sh --ngpu 8 --stats-dir ... --train-unregistered-specifier ...
#
# runs the schedule and stops after export, because the data to decode is
# not prepared here. Pick up where you left off, or run one stage, with
# --stage and --stop-stage:
#
#   ./run.sh --stage sft --stop-stage sft --ngpu 8 ...
#   ./run.sh --stage export
#
# Each training stage starts from the latest complete checkpoint of the one
# before it, and continues its own output directory once that has one, so
# repeating a command resumes an interrupted stage rather than restarting
# it. To decode without training anything, point the inference stage at
# published weights:
#
#   ./run.sh --stage infer \
#       --export-path /path/to/bagpiper-sft/model.pt \
#       --train-config /path/to/bagpiper-sft/train_stage3_qwen3_base.yaml \
#       --inference-config /path/to/bagpiper-sft/inference_audio.yaml \
#       --test-unregistered-specifier 'dialogue:test:/path/to/test.json'
#
# Training options are those of ../../TEMPLATE/speechlm1/train.sh and are
# passed straight through. See README.md, and ./run.sh --help.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ../../TEMPLATE/speechlm1/run.sh \
    --train-stages "warmup:conf/train.yaml:exp/warmup \
                    pretrain:conf/tuning/train_pretrain.yaml:exp/pretrain \
                    sft:conf/tuning/train_sft.yaml:exp/sft" \
    --wandb-project bagpiper \
    "$@"
