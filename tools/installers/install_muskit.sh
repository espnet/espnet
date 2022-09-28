#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install ParallelWaveGAN
if ! python3 -c "import parallel_wavegan.version" &> /dev/null; then
    (
        set -euo pipefail
        rm -rf ParallelWaveGAN
        git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
        cd ParallelWaveGAN
        pip install -e .
        cd ../
    )
else
    echo "parallel_wavegan is already installed"
fi

# Install pytsmod
if ! python3 -c "import pytsmod.version" &> /dev/null; then
    (
        set -euo pipefail
        python3 -m pip install pytsmod
    )
else
    echo "pytsmod is already installed"
fi

# Install miditoolkit
if ! python3 -c "import miditoolkit.version" &> /dev/null; then
    (
        set -euo pipefail
        python3 -m pip install miditoolkit
    )
else
    echo "miditoolkit is already installed"
fi

# Install music21
if ! python3 -c "import music21.version" &> /dev/null; then
    (
        set -euo pipefail
        python3 -m pip install music21
    )
else
    echo "music21 is already installed"
fi

