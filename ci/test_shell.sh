#!/usr/bin/env bash

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${GITHUB_PR_LABEL_ESPNET1:-}" ] && [ -z "${GITHUB_PR_LABEL_ESPNET2:-}" ]; then
    # If not Label tag ESPNET 1 or ESPNET 2 but Docker, then skip all tests.
    if [ -n "${GITHUB_PR_LABEL_DOCKER:-}" ]; then
        log Only Docker related modifications. Skipping tests.
        exit 0
    fi
fi

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
    wget https://github.com/koalaman/shellcheck/releases/download/stable/shellcheck-stable.linux.x86_64.tar.xz
    tar -xvf shellcheck-stable.linux.x86_64.tar.xz
fi
. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

echo "=== run shellcheck ==="
rm -fv tools/Miniconda*.sh  # exclude from schellcheck
find ci utils doc egs2/TEMPLATE/*/scripts egs2/TEMPLATE/*/setup.sh tools/*.sh -name "*.sh" -printf "=> %p\n" -execdir shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 {} \; | tee check_shellcheck
find egs2/*/*/local/data.sh -printf "=> %p\n" -execdir sh -c 'cd .. ; shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 local/$1 ; ' -- {} \; | tee -a check_shellcheck
find egs egs2 \( -name "run.sh" -o -name asr.sh -o -name tts.sh -o -name enh.sh \) -printf "=> %p\n" -execdir shellcheck -Calways -x -e SC2001 -e SC1091 -e SC2086 {} \; | tee -a check_shellcheck

if grep -q "SC[0-9]\{4\}" check_shellcheck; then
    echo "[ERROR] shellcheck failed"
    exit 1
fi

echo "=== run bats ==="
bats test_utils/test_*.bats test_utils/espnet2_scripts/test_*.bats
