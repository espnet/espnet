#!/usr/bin/env bash
set -euo pipefail

# SCTK official repo does not have version tags. Here's the mapping:
# 2.4.9 = 659bc36; 2.4.10 = d914e1b; 2.4.11 = 20159b5.
SCTK_GITHASH=20159b5

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

if [ ! -e sctk-"${SCTK_GITHASH}".tar.gz ]; then
    wget -nv -T 10 -t 3 -O sctk-"${SCTK_GITHASH}".tar.gz \
        https://github.com/usnistgov/SCTK/archive/"${SCTK_GITHASH}".tar.gz
fi

if [ ! -e sctk ]; then
    tar zxvf sctk-"${SCTK_GITHASH}".tar.gz
    rm -rf sctk-"${SCTK_GITHASH}" sctk
    mv SCTK-"${SCTK_GITHASH}"* sctk-"${SCTK_GITHASH}"
    if [[ $(uname -s) =~ MINGW || $(uname -s) =~ MSYS ]]; then
        # FIXME(kamo): For MINGW or MSYS, ln command can't work by default (it can work with CYGWIN)
        mv sctk-"${SCTK_GITHASH}" sctk
    else
        ln -s sctk-"${SCTK_GITHASH}" sctk
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
