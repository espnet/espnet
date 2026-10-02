#!/usr/bin/env bash

# Bagpiper-TTS, from a checkpoint to decoded output.
#
# One training stage, started from the published Bagpiper base weights:
#
#   ./run.sh --ngpu 8 --resume-path /path/to/bagpiper-base/base.pt \
#       --stats-dir ... --train-unregistered-specifier ...
#
# then
#
#   ./run.sh --stage export
#   ./run.sh --stage infer --inference-config inference.yaml \
#       --test-unregistered-specifier 'dialogue:test:/path/to/test.json'
#
# Omit --resume-path to continue an interrupted run: the latest complete
# checkpoint under exp/sft restores the model, optimizer and step. The data
# is yours to supply; see README.md. Training options are those of
# ../../TEMPLATE/speechlm1/train.sh and are passed straight through;
# ./run.sh --help lists the rest.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ../../TEMPLATE/speechlm1/run.sh \
    --train-config conf/train.yaml \
    --output-dir exp/sft \
    --wandb-project bagpiper-tts \
    "$@"
