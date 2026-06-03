#!/usr/bin/env bash
set -euo pipefail

SCTK_ARCHIVE=sctk-master.zip
SCTK_URL=https://github.com/usnistgov/SCTK/archive/refs/heads/master.zip
SCTK_DIR=sctk-master

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
if [[ ${unames} =~ MINGW || ${unames} =~ MSYS ]]; then
    # FIXME(kamo): CYGWIN may be okay
    echo "Warning: sctk might not be able to be built with ${unames}. Please use CYGWIN. Exit with doing nothing"
    exit 0
fi

if [ ! -e "${SCTK_ARCHIVE}" ]; then
    wget -nv -T 10 -t 3 -O "${SCTK_ARCHIVE}" "${SCTK_URL}"
fi

if [ ! -e sctk ]; then
    unzip -o "${SCTK_ARCHIVE}"
    rm -rf "${SCTK_DIR}" sctk
    mv SCTK-master "${SCTK_DIR}"
    if [[ $(uname -s) =~ MINGW || $(uname -s) =~ MSYS ]]; then
        # FIXME(kamo): For MINGW or MSYS, ln command can't work by default (it can work with CYGWIN)
        mv "${SCTK_DIR}" sctk
    else
        ln -s "${SCTK_DIR}" sctk
    fi
fi

(
    set -euo pipefail
    cd sctk
    export CFLAGS="${CFLAGS:-} -w"
    export CXXFLAGS="${CXXFLAGS:-} -std=c++11 -w"
    make config
    touch .configured
    make all && make install && make doc
)
