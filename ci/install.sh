#!/usr/bin/env bash

# NOTE: DO NOT WRITE DISTRIBUTION-SPECIFIC COMMANDS HERE (e.g., apt, dnf, etc)

set -euo pipefail

# Timer functions for measuring command execution time
start_timer() {
    local label="$1"
    echo "::group::$label"
    SECONDS=0
}

end_timer() {
    local label="$1"
    echo "::endgroup::"
    local elapsed=$SECONDS
    printf "⏱️  %s completed in %02d:%02d:%02d\n" "$label" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
}

${CXX:-g++} -v

(
    set -euo pipefail
    cd tools

    # To skip error
    mkdir -p kaldi/egs/wsj/s5/utils && touch kaldi/egs/wsj/s5/utils/parse_options.sh
    if ${USE_CONDA}; then
        ./setup_miniforge.sh venv espnet ${ESPNET_PYTHON_VERSION}
        # To install via pip instead of conda
    else
        ./setup_venv.sh "$(command -v python3)" venv
    fi

    . ./activate_python.sh
    # FIXME(kamo): Failed to compile pesq
    make TH_VERSION="${TH_VERSION}" WITH_OMP="${WITH_OMP-ON}" all \
        warp-transducer.done nkf.done moses.done mwerSegmenter.done \
        pyopenjtalk.done py3mmseg.done s3prl.done transformers.done \
        phonemizer.done fairseq.done k2.done longformer.done \
        parallel-wavegan.done muskits.done lora.done sph2pipe \
        versa.done torcheval.done whisper.done
    rm -rf kaldi
)
. tools/activate_python.sh
python3 --version

start_timer "install kenlm"
python3 -m pip install https://github.com/kpu/kenlm/archive/master.zip
end_timer "install kenlm"
# NOTE(kamo): tensorboardx is used for chainer mode only
start_timer "install tensorboardx and matplotlib"
python3 -m pip install tensorboardx
# NOTE(kamo): Create matplotlib.cache to reduce runtime for test phase
python3 -c "import matplotlib.pyplot"
end_timer "install tensorboardx and matplotlib"
# NOTE(wangyou): onnxruntime and onnx2torch are used for testing dnsmos functions
cat >> constraints.txt << EOF
torch==${TH_VERSION}
EOF
start_timer "install onnxruntime onnx2torch"
python3 -m pip install -c constraints.txt onnxruntime onnx2torch --extra-index-url https://download.pytorch.org/whl/cpu
end_timer "install onnxruntime onnx2torch"

# NOTE(kan-bayashi): Fix the error in black installation.
#   See: https://github.com/psf/black/issues/1707
python3 -m pip uninstall -y typing

# NOTE(kamo): Workaround for pip resolve issue (I think this is a bug of pip)
start_timer "install hacking flake8"
python3 -m pip install "hacking>=2.0.0" "flake8>=3.7.8"
end_timer "install hacking flake8"

# install espnet
start_timer "install espnet"
python3 -m pip install -e ".[test,doc,all]"
end_timer "install espnet"

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

# Check numpy version
python3 <<EOF
import numpy
from packaging.version import parse as L

if L(numpy.__version__) < L("2.0.0"):
    raise RuntimeError(f"Numpy>=2.0.0 is expected, but got numpy={numpy.__version__}. This is a bug in installation scripts")
EOF
