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
        ./setup_python.sh "$(command -v python3)" venv
    fi
    . ./activate_python.sh
    make TH_VERSION="${TH_VERSION}"

    make warp-ctc.done warp-transducer.done chainer_ctc.done nkf.done moses.done mwerSegmenter.done pesq pyopenjtalk.done py3mmseg.done
    rm -rf kaldi
)
. tools/activate_python.sh
python3 --version

pip3 install https://github.com/kpu/kenlm/archive/master.zip

if ${USE_CONDA}; then
  conda install -c k2-fsa -c pytorch k2 cpuonly pytorch=${TH_VERSION}
else
  pip3 install k2==${K2_VERSION}.torch${TH_VERSION} -f https://k2-fsa.org/nightly/
fi


# NOTE(kan-bayashi): Fix the error in black installation.
#   See: https://github.com/psf/black/issues/1707
pip3 uninstall -y typing

# install espnet
pip3 install -e ".[test]"
pip3 install -e ".[doc]"

# log
pip3 freeze
