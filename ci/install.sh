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
        # To install via pip instead of conda
    else
        ./setup_venv.sh "$(command -v python3)" venv
    fi

    . ./activate_python.sh
    # FIXME(kamo): Failed to compile pesq
    make TH_VERSION="${TH_VERSION}" WITH_OMP="${WITH_OMP-ON}" all warp-transducer.done chainer_ctc.done nkf.done moses.done mwerSegmenter.done pyopenjtalk.done py3mmseg.done s3prl.done transformers.done phonemizer.done fairseq.done k2.done gtn.done longformer.done whisper.done parallel-wavegan.done muskits.done
    rm -rf kaldi
)
. tools/activate_python.sh
python3 --version

python3 -m pip install https://github.com/kpu/kenlm/archive/master.zip
# NOTE(kamo): tensorboardx is used for chainer mode only
python3 -m pip install tensorboardx
# NOTE(kamo): Create matplotlib.cache to reduce runtime for test phase
python3 -c "import matplotlib.pyplot"

# NOTE(kan-bayashi): Fix the error in black installation.
#   See: https://github.com/psf/black/issues/1707
python3 -m pip uninstall -y typing

# NOTE(kamo): Workaround for pip resolve issue (I think this is a bug of pip)
python3 -m pip install "hacking>=2.0.0" "flake8>=3.7.8"

# install espnet
python3 -m pip install -e ".[test]"
python3 -m pip install -e ".[doc]"

# log
python3 -m pip freeze


# Check pytorch version
python3 <<EOF
import torch
from packaging.version import parse as L
version = '$TH_VERSION'.split(".")
next_version = f"{version[0]}.{version[1]}.{int(version[2]) + 1}"

if L(torch.__version__) < L('$TH_VERSION') or L(torch.__version__) >= L(next_version):
    raise RuntimeError(f"Pytorch=$TH_VERSION is expected, but got pytorch={torch.__version__}. This is a bug in installation scripts")
EOF
