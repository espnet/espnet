#!/usr/bin/env bash

# NOTE: DO NOT WRITE DISTRIBUTION-SPECIFIC COMMANDS HERE (e.g., apt, dnf, etc)

set -euo pipefail

if [[ ${USE_CONDA} == false ]]; then
    if [[ ${TH_VERSION} == nightly ]]; then
        pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
    else
        pip install --quiet torch=="${TH_VERSION}" -f https://download.pytorch.org/whl/cpu/stable
    fi
else
    (
        cd tools
        make PYTHON_VERSION=${ESPNET_PYTHON_VERSION} venv
    )
    . tools/venv/etc/profile.d/conda.sh
    conda config --set always_yes yes
    conda activate
    conda update -y conda
    if [[ ${TH_VERSION} == nightly ]]; then
        conda install pytorch-nightly-cpu -c pytorch
    else
        conda install -y pytorch-cpu="${TH_VERSION}" -c pytorch
    fi
fi

python --version

pip install -U pip wheel
pip install chainer=="${CHAINER_VERSION}"

# install espnet
pip install -e .
pip install -e ".[test]"
pip install -e ".[doc]"

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
pip install -U flake8

# install matplotlib
pip install matplotlib

# install warp-ctc (use @jnishi patched version)
git clone https://github.com/jnishi/warp-ctc.git -b pytorch-1.0.0
cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 && cd ..
pip install cffi
cd pytorch_binding && python setup.py install && cd ../..

# install kaldiio
pip install git+https://github.com/nttcslab-sp/kaldiio.git

# install chainer_ctc
pip install cython
git clone https://github.com/jheymann85/chainer_ctc.git
cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh
pip install . && cd ..

# log
pip freeze
