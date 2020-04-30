#!/usr/bin/env bash

# NOTE: DO NOT WRITE DISTRIBUTION-SPECIFIC COMMANDS HERE (e.g., apt, dnf, etc)

set -euo pipefail

$CXX -v

if ${USE_CONDA}; then
    (
        cd tools
        make PYTHON_VERSION=${ESPNET_PYTHON_VERSION} venv
    )
    . tools/venv/etc/profile.d/conda.sh
    conda config --set always_yes yes
    conda activate
    conda update -y conda
    if [[ ${TH_VERSION} == nightly ]]; then
        conda install -q -y pytorch-nightly-cpu -c pytorch
    else
        conda install -q -y pytorch="${TH_VERSION}" cpuonly -c pytorch
    fi
    conda install -c conda-forge ffmpeg
else
    # to suppress errors during doc generation of utils/ when USE_CONDA=false in travis
    mkdir -p tools/venv/bin
    touch tools/venv/bin/activate
    . tools/venv/bin/activate

    if [[ ${TH_VERSION} == nightly ]]; then
        pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
    elif [[ ${TH_VERSION} == 1.0.1 ]] || [[ ${TH_VERSION} == 1.1.0 ]]; then
        pip install --quiet torch=="${TH_VERSION}" -f https://download.pytorch.org/whl/cpu/stable
    else
        pip install --quiet torch=="${TH_VERSION}+cpu" -f https://download.pytorch.org/whl/torch_stable.html
    fi
fi

python --version

pip install -U wheel
# Fix pip version to avoid this error https://github.com/ethereum/eth-abi/issues/131#issuecomment-620981271
pip install pip==20.0.2
pip install chainer=="${CHAINER_VERSION}"

# install espnet
pip install -e .
pip install -e ".[test]"
pip install -e ".[doc]"

# [FIXME] hacking==1.1.0 requires flake8<2.7.0,>=2.6.0, but that version has a problem around fstring
pip install -U flake8 flake8-docstrings

# install matplotlib
pip install matplotlib

# install warp-ctc (use @jnishi patched version)
git clone https://github.com/jnishi/warp-ctc.git -b pytorch-1.0.0
cd warp-ctc && mkdir build && cd build && cmake .. && make -j4 && cd ..
pip install cffi
cd pytorch_binding && python setup.py install && cd ../..

# install chainer_ctc
pip install cython
mkdir -p tools
cd tools && git clone https://github.com/jheymann85/chainer_ctc.git
cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh
pip install . && cd ../..

# install warp-transducer
git clone https://github.com/HawkAaron/warp-transducer.git
cd warp-transducer && mkdir build && cd build && cmake .. && make && cd ..
cd pytorch_binding && python setup.py install && cd ../..

# log
pip freeze
