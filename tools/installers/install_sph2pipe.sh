#!/usr/bin/env bash
set -euo pipefail

SPH2PIPE_VERSION=2.5

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
if [[ ${unames} =~ MINGW || ${unames} =~ MSYS ]]; then
    # FIXME(kamo): CYGWIN may be okay
    echo "Warning: sph2pipe might not be able to be built with ${unames}. Please use CYGWIN. Exit with doing nothing"
    exit 0
fi

if [ ! -e sph2pipe-${SPH2PIPE_VERSION}.tar.gz ]; then
    wget -nv -T 10 -t 3 -O sph2pipe-${SPH2PIPE_VERSION}.tar.gz \
	    https://github.com/burrmill/sph2pipe/archive/${SPH2PIPE_VERSION}.tar.gz
fi

if [ ! -e sph2pipe-${SPH2PIPE_VERSION} ]; then
    tar --no-same-owner -xzf sph2pipe-${SPH2PIPE_VERSION}.tar.gz
    rm -rf sph2pipe && ln -s  sph2pipe-${SPH2PIPE_VERSION} sph2pipe
fi

make -C sph2pipe
