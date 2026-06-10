#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

{
    python3 -m pip install flash-attn --no-build-isolation
} || {
    echo "Flash Attention failed to install, trying without building CUDA"
    MAX_JOBS=16 python3 -m pip install flash-attn --no-build-isolation
} || {
    echo "Failed to install flash attention. Skipping..."
    echo "Manual install may be required: https://github.com/Dao-AILab/flash-attention"
}
