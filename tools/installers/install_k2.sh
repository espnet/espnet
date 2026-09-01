#!/usr/bin/env bash
set -euo pipefail

# k2 publishes one wheel per (k2 version, torch version, python version), indexed
# at https://k2-fsa.github.io/k2/cpu.html and .../cuda.html. The files themselves
# live on Hugging Face under csukuangfj2/k2, which is the maintainer's account -
# that is k2's documented distribution channel, not a mirror of our choosing.
#
# 1.24.4.dev20260625 is the one release that covers every python x torch pair
# ci/image_variants.json builds: cp312 and cp313 against torch 2.9.1, 2.10.0 and
# 2.11.0. Check the index before adding a torch version, because k2 lags torch.
pip_k2_version="1.24.4.dev20260625"

# The conda channel stops at 1.24.3.dev20230508 and has nothing for torch 2.9 or
# later, so the conda path cannot install k2 at all here. Left empty deliberately;
# the branch below skips with a reason rather than pretending.
conda_k2_version=""

if [ $# -gt 2 ]; then
    echo "Usage: $0 [use-conda|true or false] [<k2-version>]"
    exit 1;
elif [ $# -gt 0 ]; then
    use_conda="$1"
    if [ "${use_conda}" != false ] && [ "${use_conda}" != true ]; then
        echo "[ERROR] <use_conda> must be true or false, but ${use_conda} is given."
        echo "Usage: $0 [use-conda|true or false] [<k2-version>]"
        exit 1
    fi

    if [ $# -eq 2 ]; then
        k2_version="$2"
        pip_k2_version="${k2_version}"
        conda_k2_version="${k2_version}"
    fi
else
    use_conda=$([[ $(conda list -e -c -f --no-pip pytorch 2>/dev/null) =~ pytorch ]] && echo true || echo false)
fi

if [[ ! $(uname -s) =~ Linux ]]; then
    echo "Warning: This script doesn't support MacOS and Windows. Please install k2 manually."
    exit 0
fi


if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi

python_36_plus=$(python3 <<EOF
from packaging.version import parse as V
import sys

if V("{}.{}.{}".format(*sys.version_info[:3])) >= V("3.6"):
    print("true")
else:
    print("false")
EOF
)

cuda_version=$(python3 <<EOF
try:
    import torch
except:
    raise RuntimeError("Please install torch before running this script")

if torch.cuda.is_available():
    version=torch.version.cuda.split(".")
    # 10.1.aa -> 10.1
    print(version[0] + "." + version[1])
else:
    print("")
EOF
)
torch_version=$(python3 <<EOF
import torch
# e.g. 1.10.0+cpu -> 1.10.0
torch_version=torch.__version__.split("+")[0]
print(torch_version)
EOF
)
libc_version="$(ldd --version | awk 'NR==1 {print $NF}')"

pytorch_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
libc_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$libc_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

echo "[INFO] torch_version=${torch_version}"
echo "[INFO] cuda_version=${cuda_version}"
echo "[INFO] libc_version=${libc_version}"

if ! "${python_36_plus}"; then
    echo "[ERROR] k2 requires python>=3.6"
    exit 1
fi

# GLIBC floor, from the wheel tags: manylinux_2_27 / manylinux_2_28.
if ! $(libc_plus 2.27); then
    echo "[WARNING] k2 wheels are manylinux_2_27, but your GLIBC is ${libc_version}. Skip k2-installation"
    exit
fi

# The pytorch-version and CUDA-version ladders that used to sit here described
# what k2 1.10 offered in 2021, keyed on that exact version string, so bumping
# the version made every one of them dead code. They are gone: the index is the
# authority on what exists, and pip consults it directly. If a wheel is missing
# for the requested combination the install fails and says which one, which is
# the outcome we want - k2 silently absent is how espnet went four years without
# noticing that the use_k2 tests never ran.

if "${use_conda}"; then
    if [ -z "${conda_k2_version}" ]; then
        echo "[WARNING] The k2-fsa conda channel stops at 1.24.3.dev20230508 and has"
        echo "[WARNING] nothing for pytorch=${torch_version}. Skip k2-installation."
        echo "[WARNING] Use the pip path (USE_CONDA=false) if you need k2."
        exit
    fi
    k2="k2=${conda_k2_version}"

    if [ -z "${cuda_version}" ]; then
        echo conda install -y -c k2-fsa -c pytorch cpuonly "${k2}" "pytorch=${torch_version}"
        conda install -y -c k2-fsa -c pytorch cpuonly "${k2}" "pytorch=${torch_version}"
    else
        # NOTE(kamo): K2 requires cudatoolkit from conda-forge channel and k2-cpu is installed if the other channel is used, e.g. anaconda, nvidia
        echo conda install -y -c k2-fsa -c pytorch -c conda-forge "${k2}" "cudatoolkit=${cuda_version}" "pytorch=${torch_version}"
        conda install -y -c k2-fsa -c pytorch -c conda-forge "${k2}" "cudatoolkit=${cuda_version}" "pytorch=${torch_version}"
    fi

else
    # https://k2-fsa.org/nightly/ - which this used to pass to -f - now 404s, so
    # even bumping the version alone would have installed nothing.
    if [ -z "${cuda_version}" ]; then
        spec="k2==${pip_k2_version}+cpu.torch${torch_version}"
        index=https://k2-fsa.github.io/k2/cpu.html
    else
        spec="k2==${pip_k2_version}+cuda${cuda_version}.torch${torch_version}"
        index=https://k2-fsa.github.io/k2/cuda.html
    fi
    # python3 -m pip, not pip: on python 3.13 `ensurepip --upgrade` leaves no
    # venv/bin/pip, so a bare `pip` falls through to the system interpreter on
    # PATH. That is not theoretical - the first version of this change installed
    # k2 into /usr/local/lib/python3.13/site-packages, which the image's final
    # stage then discarded, while the log said "Successfully installed k2".
    #
    # --no-deps because the wheel declares torch==2.9.1, and resolving that pulls
    # the default PyPI torch, which is the CUDA build: ~3 GB of nvidia-* wheels
    # replacing the CPU torch this environment installed on purpose. k2's only
    # other dependency is graphviz, installed explicitly below.
    echo python3 -m pip install --no-deps "${spec}" -f "${index}"
    python3 -m pip install --no-deps "${spec}" -f "${index}" || {
        echo "[ERROR] No k2 wheel for ${spec}" >&2
        echo "[ERROR] Check ${index} - k2 publishes per torch and python version," >&2
        echo "[ERROR] and lags new torch releases. Either pick a k2 version that" >&2
        echo "[ERROR] covers every pair in ci/image_variants.json, or drop k2.done" >&2
        echo "[ERROR] from ci/install.sh rather than let it install nothing." >&2
        exit 1
    }
    python3 -m pip install graphviz

    # Prove it, because the install saying "Successfully installed k2" did not
    # mean k2 was importable - see above.
    python3 -c "import k2; print('k2', k2.__version__, 'from', k2.__file__)" || {
        echo "[ERROR] k2 installed but does not import." >&2
        echo "[ERROR] Check which interpreter it landed in:" >&2
        echo "[ERROR]   python3 -c 'import sys; print(sys.executable)'" >&2
        exit 1
    }
fi
