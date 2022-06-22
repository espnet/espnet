#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

rm -rf muskit.done
rm -rf ParallelWaveGAN
git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
cd ParallelWaveGAN
pip install -e .
cd ../
touch muskit.done
