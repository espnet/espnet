#!/usr/bin/env bash

set -euo pipefail

# This is the last stable version. MFA 3.0 depends on an unstable kaldi version that creates errors
conda install -y -c conda-forge montreal-forced-aligner=2.2.17

# Installing with pip is not recommended as many dependencies are missing!
# Also, baumwelch is only available on conda-forge. If you REALLY want to do it:
# conda install -c conda-forge baumwelch  # includes openfst
# conda install postgresql
# pip install pgvector pynini hdbscan
# git clone https://github.com/kaldi-asr/kaldi.git
#     (
#         set -euo pipefail
#         sudo apt-get install zlib1g-dev gfortran subversion python2.7 intel-mkl
#         cd kaldi/tools && make -j 4
#         x=""; for b in $(ls | grep bin); do x="$x:$(pwd)/$b"; done; export PATH="$PATH:$x"
#     )  # install more dependencies if make doesn't work
#     (
#        set -euo pipefail
#        cd kaldi/src
#        ./configure --shared
#        make clean depend
#        make -j 8
#        x=""; for b in $(ls | grep bin); do x="$x:$(pwd)/$b"; done; export PATH="$PATH:$x"
#     )
# pip install montreal-forced-aligner
