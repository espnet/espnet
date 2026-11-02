#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    echo "Usage: $0 [with_openmp]"
    echo "  with_openmp: ON or OFF (default: ON)"
}

if [ $# -eq 0 ]; then
    with_openmp="ON"
elif [ $# -eq 1 ]; then
    if [[ "$1" == "ON" || "$1" == "OFF" ]]; then
        with_openmp=$1
    else
        print_usage
        echo "Invalid argument: $1. Must be ON or OFF"
        exit 1
    fi
else
    print_usage
    echo "Too many arguments"
    exit 1
fi

unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi

rm -rf warp-transducer
git clone https://github.com/ljn7/warp-transducer.git

(
    set -euo pipefail
    cd warp-transducer

    mkdir build
    (
        set -euo pipefail
        cd build && cmake -DWITH_OMP="${with_openmp}" .. && make
        # cd build && cmake -DWITH_OMP="${with_openmp}" -DCMAKE_CXX_FLAGS="-std=c++1z" .. && make

    )

    (
        set -euo pipefail
        # --use-pep517 as well as --no-build-isolation. Without it, a project
        # with only a setup.py takes pip's legacy editable path, `setup.py
        # develop` - and setuptools>=80 has turned that command into a wrapper
        # that re-invokes `pip install -e . --use-pep517 --no-deps` on its own,
        # in a fresh isolated environment. The isolation this line asks for is
        # lost there, so this setup.py's `import torch` fails with
        # ModuleNotFoundError and the whole install dies. With --use-pep517 the
        # build runs in this environment, where torch is, and no legacy command
        # is involved.
        # The binding hardcodes the C++ standard - "-std=c++17" for torch>=2.1
        # - and torch's own extension builder fills one in only when none is
        # given (append_std17_if_no_std_present, which despite the name appends
        # -std=c++20 now). The binding's flag lands last on the command line, so
        # it wins, and torch 2.14's headers use C++20 `requires` clauses:
        # compiled at C++17 the build dies with "'requires' does not name a
        # type" in ATen/core/TensorBase.h. Take the flag out and let torch
        # choose the standard its own headers need, this version and the next.
        python3 - <<'PATCH'
import pathlib, re

path = pathlib.Path("pytorch_binding/setup.py")
before = path.read_text()
after, replaced = re.subn(
    r"extra_compile_args \+= \['-std=c\+\+\d+'\]",
    "pass  # espnet: let torch.utils.cpp_extension pick the standard",
    before,
)
if not replaced:
    raise SystemExit(
        "warp-transducer's setup.py no longer sets -std=; check whether it now "
        "leaves the standard to torch, and drop this patch if so"
    )
path.write_text(after)
PATCH
        cd pytorch_binding && python3 -m pip install --use-pep517 --no-build-isolation -e .
    )
)

if ! python3 -c "import ninja" &> /dev/null; then
    (
        set -euo pipefail
        echo "Installing ninja package for RWKV decoder (training only)."

        python3 -m pip install ninja
    )
fi
