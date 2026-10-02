#!/usr/bin/env bash

# Bagpiper-TTS, in order: SFT, export, inference.
#
#   ./run.sh --ngpu 8 --resume-path /path/to/bagpiper-base/base.pt \
#       --stats-dir ... --train-unregistered-specifier ...
#
# trains and exports; the data to decode is not prepared here, so the
# inference stage is asked for separately. There is one training stage, so
# it needs starting weights: the published Bagpiper-Base `base.pt`. Repeat
# the command without --resume-path to continue an interrupted run - the
# latest complete checkpoint under exp/sft restores the model, optimizer
# and step.
#
#   ./run.sh --stage export
#   ./run.sh --stage infer \
#       --inference-config /path/to/bagpiper-tts-sft/inference.yaml \
#       --test-unregistered-specifier 'dialogue:test:/path/to/test.json'
#
# To decode published weights without training, add --export-path and
# --train-config pointing into the downloaded model directory.
#
# Training options are those of ../../TEMPLATE/speechlm1/train.sh and are
# passed straight through. See README.md, and ./run.sh --help.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ../../TEMPLATE/speechlm1/run.sh \
    --train-stages "sft:conf/train.yaml:exp/sft" \
    --wandb-project bagpiper-tts \
    "$@"
