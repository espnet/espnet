#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

rm -rf ParallelWaveGAN

# ParallelWaveGAN  Commit id when making this PR: `commit 4615144d75bcb519ff1d2df7699ddd626787b5a4`
git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
cd ParallelWaveGAN
pip install -e .
cd ..
