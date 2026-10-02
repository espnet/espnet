#!/usr/bin/env bash

# Bagpiper, from a checkpoint to decoded output.
#
# This recipe has three stages, and runs training alone by default: the
# data it decodes is not prepared here, so a full pass needs a test
# manifest you provide.
#
#   ./run.sh --ngpu 8 --stats-dir ... --train-unregistered-specifier ...
#   ./run.sh --stage export
#   ./run.sh --stage infer --inference-config inference_audio.yaml \
#       --test-unregistered-specifier 'dialogue:test:/path/to/test.json'
#
# Training options are those of ../../TEMPLATE/speechlm1/train.sh and are
# passed straight through. See README.md, and --help for the stages.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ../../TEMPLATE/speechlm1/run.sh \
    --train-config conf/train.yaml \
    --output-dir exp/warmup \
    --wandb-project bagpiper \
    "$@"
