#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

unames="$(uname -s)"

print_usage() {
    echo "Usage: $0 [with_openmp]"
    echo "  with_openmp: ON or OFF (default: ON)"
}

error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}Warning: $1${NC}" >&2
}

info() {
    echo -e "${GREEN}$1${NC}"
}

check_requirements() {
    local requirements=("git" "cmake" "python3" "gcc")
    for cmd in "${requirements[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "$cmd is required but not installed"
        fi
    done
}

check_args() {
    if [ $# -eq 0 ]; then
        with_openmp="ON"
    elif [ $# -eq 1 ]; then
        if [[ "$1" == "ON" || "$1" == "OFF" ]]; then
            with_openmp=$1
        else
            print_usage
            error_exit "Invalid argument: $1. Must be ON or OFF"
        fi
    else
        print_usage
        error_exit "Too many arguments"
    fi
}

check_platform() {
    case "$unames" in
        Linux|Darwin) ;;
        *) info "Platform $unames is not supported. Exiting cleanly."
           exit 0 ;;
    esac
}

ensure_python_module() {
    local module=$1
    local pip_name=${2:-$module}
    if ! python3 -c "import $module" &> /dev/null; then
        info "Installing missing Python module: $pip_name"
        if ! python3 -m pip install "$pip_name"; then
            error_exit "Could not install $pip_name"
        fi
    fi
}

check_gcc_version() {
    info "Checking GCC version compatibility..."
    python3 - <<'EOF' || warn "GCC version check failed (requires >= 4.9.2)"
from packaging.version import parse as V
import subprocess
try:
    gcc_version = subprocess.check_output(['gcc', '-dumpversion'], text=True).strip()
    if V(gcc_version) < V('4.9.2'):
        print(f"GCC version {gcc_version} is too old. Requires >= 4.9.2")
        exit(1)
except Exception as e:
    print(f"Could not verify GCC version: {e}")
    exit(1)
EOF
}

clone_repo() {
    if [ -d "warp-transducer" ]; then
        info "Found existing warp-transducer directory, removing..."
        rm -rf warp-transducer || error_exit "Failed to remove existing directory"
    fi

    info "Cloning warp-transducer repository..."
    git clone https://github.com/ljn7/warp-transducer.git || error_exit "Failed to clone warp-transducer"
}

setup_cuda_env() {
    if [[ "$unames" != "Linux" ]]; then
        info "Skipping CUDA setup on $unames"
        return
    fi

    : "${CUDA_HOME:=/usr/local/cuda}"
    if [[ ! -d "$CUDA_HOME" ]]; then
        warn "CUDA not found at $CUDA_HOME - skipping CUDA setup"
        return
    fi

    arch="$(uname -m)"
    local cuda_include=""
    local cuda_lib=""

    case "$arch" in
        x86_64)
            cuda_include="$CUDA_HOME/targets/x86_64-linux/include"
            cuda_lib="$CUDA_HOME/targets/x86_64-linux/lib"
            ;;
        aarch64)
            cuda_include="$CUDA_HOME/include"
            cuda_lib="$CUDA_HOME/lib64"
            ;;
        *)
            warn "Unknown architecture $arch. Skipping CUDA path export."
            return
            ;;
    esac

    if [[ -d "$cuda_include" && -d "$cuda_lib" ]]; then
        info "Configuring CUDA environment for $arch"
        export CFLAGS="${CFLAGS:-} -I$cuda_include"
        export CXXFLAGS="${CXXFLAGS:-} -I$cuda_include"
        export LDFLAGS="${LDFLAGS:-} -L$cuda_lib"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_lib:${LD_LIBRARY_PATH:-}"
    else
        warn "CUDA include/lib not found at expected paths: $cuda_include, $cuda_lib"
    fi
}

build_project() {
    info "Starting CMake build (OpenMP=$with_openmp)..."
    mkdir -p build || error_exit "Failed to create build directory"
    cd build || error_exit "Failed to enter build directory"
    cmake -DWITH_OMP="$with_openmp" .. || error_exit "CMake configuration failed"
    make || error_exit "Build failed"
    cd ..
}

install_python_binding() {
    cd pytorch_binding || error_exit "Failed to enter pytorch_binding directory"
    info "Installing Python binding..."
    python3 -m pip install -e . || error_exit "Failed to install Python binding"
    cd ..
}

check_optional_tools() {
    if ! python3 -c "import ninja" &>/dev/null; then
        info "Installing optional 'ninja' package for RWKV decoder support..."
        python3 -m pip install ninja || warn "ninja installation failed (not critical)"
    else
        info "ninja is already installed"
    fi
}

main() {
    check_args "$@"
    check_requirements
    check_platform
    ensure_python_module packaging
    check_gcc_version
    clone_repo

    pushd warp-transducer > /dev/null || error_exit "Failed to enter warp-transducer directory"
    setup_cuda_env
    build_project
    install_python_binding
    popd > /dev/null || error_exit "Failed to return to original directory"

    check_optional_tools

    info "warp-transducer installation completed successfully!"
}

main "$@"
