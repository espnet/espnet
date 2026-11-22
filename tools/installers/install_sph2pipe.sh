#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/installers/download_with_retry.sh
. "$(dirname "$0")"/download_with_retry.sh

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
    download_with_retry \
        https://github.com/burrmill/sph2pipe/archive/${SPH2PIPE_VERSION}.tar.gz \
        sph2pipe-${SPH2PIPE_VERSION}.tar.gz -nv -T 10
fi

if [ ! -e sph2pipe-${SPH2PIPE_VERSION} ]; then
    tar --no-same-owner -xzf sph2pipe-${SPH2PIPE_VERSION}.tar.gz
    rm -rf sph2pipe && ln -s  sph2pipe-${SPH2PIPE_VERSION} sph2pipe
fi

# sph2pipe 2.5 is K&R C: "ulong word_get()" declares no prototype and strdup
# is used without <string.h>. The GCC of the ubuntu-26.04 CI image compiles it
# under C23 rules, where "()" means "no arguments" (word_get(fpin) is an error),
# and recent clang rejects the implicit strdup declaration. Compile it as the C
# it was written in.
make -C sph2pipe CCFLAGS="${CCFLAGS:-} -std=gnu89"
