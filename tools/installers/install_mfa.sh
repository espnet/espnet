#!/usr/bin/env bash

set -euo pipefail

if [ $# != 1 ]; then
    echo "Usage: $0 true/false"
    exit 1;
fi

use_conda="$1"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
python_major=$(python3 -c 'import sys; print(sys.version_info[0])')
python_minor=$(python3 -c 'import sys; print(sys.version_info[1])')

if [ "$python_major" -gt 3 ] || [ "$python_major" -eq 3 -a "$python_minor" -ge 11 ]; then
    echo "WARNING: Python version $python_version detected."
    echo "Python versions >= 3.11 do not include 'imp' module."
    echo "You may need to downgrade Python to 3.10 or add a package that emulates 'imp'."
fi

if "${use_conda}"; then
    conda install -c conda-forge pynini -y
    conda install -c conda-forge ngram  -y # for training g2p
    conda install -c conda-forge baumwelch -y # for training g2p
else
    echo "Error: MFA dependencies (e.g., pynini) must be installed via conda. Please run with use_conda=true."
    exit 1
    #TODO(fhrozen): review the required packages on pip
    pip install --only-binary :all: pynini
fi

# Use only pip, conda installation may cause issues due to version match.
pip install sqlalchemy==1.4.45  # v2.0.0+ generates error on MFA
pip install --ignore-requires-python git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git@v2.0.6
