#!/usr/bin/env bash
set -e
set -u
set -o pipefail

. ./sv.sh \
    --n_gpu 4 \
    --n_train_frame 300 \
    --n_eval_frame 400
