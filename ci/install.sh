#!/usr/bin/env bash

set -euo pipefail

pip install -U pip wheel

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
