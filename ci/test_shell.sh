#!/usr/bin/env bash

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

PATH=$(pwd)/bats-core/bin:$(pwd)/shellcheck-stable:$PATH
if ! [ -x "$(command -v bats)" ]; then
    echo "=== install bats ==="
    git clone https://github.com/bats-core/bats-core.git
fi
if ! [ -x "$(command -v shellcheck)" ]; then
    echo "=== install shellcheck ==="
    wget https://storage.googleapis.com/shellcheck/shellcheck-stable.linux.x86_64.tar.xz
    tar -xvf shellcheck-stable.linux.x86_64.tar.xz
fi
if ${USE_CONDA:-}; then
    . tools/venv/bin/activate
fi

set -euo pipefail

echo "=== run shellcheck ==="
find ci utils doc egs2/TEMPLATE/*/scripts egs2/TEMPLATE/*/setup.sh -name "*.sh" -printf "=> %p\n" -execdir shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 {} \; | tee check_shellcheck
find egs2/*/*/local/data.sh -printf "=> %p\n" -execdir sh -c 'cd .. ; shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 local/$1 ; ' -- {} \; | tee check_shellcheck
find egs egs2 \( -name "run.sh" -o -name asr.sh -o -name tts.sh \) -printf "=> %p\n" -execdir shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 {} \; | tee -a check_shellcheck

if grep -q "SC[0-9]\{4\}" check_shellcheck; then
    echo "[ERROR] shellcheck failed"
    exit 1
fi

echo "=== run bats ==="
bats test_utils/test_*.bats
