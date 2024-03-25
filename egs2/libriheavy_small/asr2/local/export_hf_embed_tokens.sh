#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;

model_name=$1
output_path=$2

if [ ! -f "${output_path}" ]; then
    local/export_hf_embed_tokens.py ${model_name} ${output_path}
fi
