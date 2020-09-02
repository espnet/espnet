#!/usr/bin/env bash

# NOTE: DO NOT WRITE DISTRIBUTION-SPECIFIC COMMANDS HERE (e.g., apt, dnf, etc)

set -euo pipefail

${CXX:-g++} -v

(
    set -euo pipefail
    cd tools

    # To skip error
    mkdir -p kaldi/egs/wsj/s5/utils && touch kaldi/egs/wsj/s5/utils/parse_options.sh
    if ${USE_CONDA}; then
        ./setup_anaconda.sh venv espnet ${ESPNET_PYTHON_VERSION}
    else
        ./setup_python.sh $(which python3) venv
    fi
    make TH_VERSION="${TH_VERSION}"

    make nkf.done moses.done mwerSegmenter.done pesq pyopenjtalk.done
    rm -rf kaldi
)
. tools/activate_python.sh
python3 --version

pip3 install https://github.com/kpu/kenlm/archive/master.zip

# install espnet
pip3 install -e ".[test]"
pip3 install -e ".[doc]"

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
pip3 install -U flake8 flake8-docstrings

# log
pip3 freeze
