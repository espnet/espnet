#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    echo "Usage: $0 [with_openmp]"
    echo "  with_openmp: ON or OFF (default: ON)"
}

if [ $# -eq 0 ]; then
    with_openmp=ON
elif [ $# -eq 1 ]; then
    if [[ "$1" == "ON" || "$1" == "OFF" ]]; then
        with_openmp=$1
    else
        echo "Error: with_openmp must be ON or OFF, got: $1"
        print_usage
        exit 1
    fi
else
    echo "Error: Too many arguments"
    print_usage
    exit 1
fi

unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exiting with no action taken."
    exit 0
fi

if ! python3 -c "import packaging.version" &> /dev/null; then
    echo "Installing packaging module..."
    python3 -m pip install packaging
fi

# Check GCC version if PyTorch is available
# Note: Removed PyTorch version restriction to handle both 1.x and 2.x
if python3 -c 'import torch' &> /dev/null; then
    echo "Checking GCC version compatibility..."
    gcc_check_script="
from packaging.version import parse as V
import subprocess
try:
    gcc_version = subprocess.check_output(['gcc', '-dumpversion'], text=True).strip()
    if V(gcc_version) < V('4.9.2'):
        print(f'GCC version {gcc_version} is too old. Requires gcc>=4.9.2')
        exit(1)
except Exception as e:
    print(f'Could not check GCC version: {e}')
    exit(1)
"
    if ! python3 -c "$gcc_check_script" 2>/dev/null; then
        echo "Warning: GCC version may be incompatible or not found"
    fi
fi

rm -rf warp-transducer

echo "Cloning warp-transducer repository..."
if ! git clone https://github.com/ljn7/warp-transducer.git; then
    echo "Error: Failed to clone repository. Check network connection and repository availability."
    exit 1
fi

(
    set -euo pipefail
    cd warp-transducer

    echo "Setting up build environment..."

    if [[ "$unames" == "Linux" ]]; then
        : "${CUDA_HOME:=/usr/local/cuda}"
        arch="$(uname -m)"

        cuda_include=""
        cuda_lib=""
        cuda_bin="$CUDA_HOME/bin"

        if [[ "$arch" == "x86_64" ]]; then
            cuda_include="$CUDA_HOME/targets/x86_64-linux/include"
            cuda_lib="$CUDA_HOME/targets/x86_64-linux/lib"
        elif [[ "$arch" == "aarch64" ]]; then
            cuda_include="$CUDA_HOME/include"
            cuda_lib="$CUDA_HOME/lib64"
        else
            echo "Warning: Unknown Linux architecture ($arch). Skipping CUDA setup."
        fi

        if [[ -n "$cuda_include" && -d "$cuda_include" && -n "$cuda_lib" && -d "$cuda_lib" ]]; then
            echo "Setting up CUDA environment for $arch..."
            export CFLAGS="${CFLAGS:-} -I\"$cuda_include\""
            export CXXFLAGS="${CXXFLAGS:-} -I\"$cuda_include\""
            export LDFLAGS="${LDFLAGS:-} -L\"$cuda_lib\""
            export PATH="$cuda_bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_lib:${LD_LIBRARY_PATH:-}"
            echo "CUDA environment configured successfully"
        else
            echo "Warning: CUDA paths not found or unsupported architecture: $arch"
            echo "  Expected include: $cuda_include"
            echo "  Expected lib: $cuda_lib"
        fi
    else
        echo "Note: Skipping CUDA setup on non-Linux platform: $unames"
    fi

    mkdir -p build

    echo "Building with CMake (OpenMP: $with_openmp)..."
    (
        set -euo pipefail
        cd build
        if ! cmake -DWITH_OMP="$with_openmp" ..; then
            echo "Error: CMake configuration failed"
            exit 1
        fi

        if ! make; then
            echo "Error: Build failed"
            exit 1
        fi
    )

    echo "Installing Python binding..."
    (
        set -euo pipefail
        cd pytorch_binding
        if ! python3 -m pip install -e .; then
            echo "Error: Failed to install Python binding"
            exit 1
        fi
    )

    echo "warp-transducer installation completed successfully"
)

if ! python3 -c "import ninja" &> /dev/null; then
    echo "Installing ninja package for RWKV decoder (training only)..."
    if ! python3 -m pip install ninja; then
        echo "Warning: Failed to install ninja package"
    else
        echo "ninja package installed successfully"
    fi
else
    echo "ninja package already installed"
fi
