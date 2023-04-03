#!/usr/bin/env bash

set -euo pipefail

if [ $# != 1 ]; then
    echo "Usage: $0 true/false"
    exit 1;
fi

use_conda="$1"

if "${use_conda}"; then
    conda install -c conda-forge pynini -y
    conda install -c conda-forge ngram  -y # for training g2p
    conda install -c conda-forge baumwelch -y # for training g2p
else
    #TODO(fhrozen): review the required packages on pip
    pip install --only-binary :all: pynini
fi

# Use only pip, conda installation may cause issues due to version match.
pip install sqlalchemy==1.4.45  # v2.0.0+ generates error on MFA
pip install --ignore-requires-python git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git@v2.0.6
