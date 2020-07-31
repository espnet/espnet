#!/usr/bin/env bash

# NOTE: DO NOT WRITE DISTRIBUTION-SPECIFIC COMMANDS HERE (e.g., apt, dnf, etc)

set -euo pipefail

$CXX -v

(
    set -euo pipefail
    cd tools
    # To suppress the installation for Kaldi
    touch kaldi.done
    if ${USE_CONDA}; then
        make PYTHON_VERSION="${ESPNET_PYTHON_VERSION}" TH_VERSION="${TH_VERSION}"
    else
        make PYTHON="$(which python)" TH_VERSION="${TH_VERSION}"
    fi
    make moses.done
    rm kaldi.done
)
. tools/activate_python.sh
python3 --version

pip3 install -U wheel
# Fix pip version to avoid this error https://github.com/ethereum/eth-abi/issues/131#issuecomment-620981271
pip3 install pip==20.0.2
pip3 install chainer=="${CHAINER_VERSION}"
pip3 install https://github.com/kpu/kenlm/archive/master.zip

# install espnet
pip3 install -e ".[test]"
pip3 install -e ".[doc]"

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
pip3 install -U flake8 flake8-docstrings

# log
pip3 freeze
