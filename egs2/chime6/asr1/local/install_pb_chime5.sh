#!/usr/bin/env bash

# Installs pb_chime5
git clone https://github.com/fgnt/pb_chime5.git
cd pb_chime5

# Download submodule dependencies  # https://stackoverflow.com/a/3796947/5766934
git submodule init
git submodule update

# sudo apt install libopenmpi-dev -- if you have problem with mpi4py installation

python -m pip install cython
python -m pip install pymongo
python -m pip install fire
python -m pip install mpi4py
python -m pip install -e pb_bss/
python -m pip install -e .
