#!/bin/bash

set -euo pipefail

if [ $# != 1 ]; then
    echo "Usage: $0 <cuda-version>"
    echo "e.g.: $0 10.0"
    exit 1;
fi

cuda_ver=$1

if [ $cuda_ver != "10.0" ] || [ $cuda_ver != "10.1" ] || [ $cuda_ver != "10.2" ]; then
    echo "warp-rnnt was not tested with CUDA_VERSION=$cuda_ver. Skipping install."
    exit 0
fi

pip install warp_rnnt
