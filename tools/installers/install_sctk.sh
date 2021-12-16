#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi


if [ ! -e sctk-2.4.10-20151007-1312Z.tar.bz2 ]; then
    wget -T 10 https://github.com/espnet/kaldi-bin/releases/download/v0.0.2/sctk-2.4.10-20151007-1312Z.tar.bz2 || \
	wget -T 10 -t 3 ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.10-20151007-1312Z.tar.bz2|| \
        wget --no-check-certificate -T 10 http://www.openslr.org/resources/4/sctk-2.4.10-20151007-1312Z.tar.bz2
fi

if [ ! -e sctk ]; then
    tar xojf sctk-2.4.10-20151007-1312Z.tar.bz2 || \
      tar --exclude '*NONE*html' -xvojf sctk-2.4.10-20151007-1312Z.tar.bz2
    rm -rf sctk && ln -s sctk-2.4.10 sctk
fi

(
    set -euo pipefail
    cd sctk
    make config
    touch .configured
    make all && make install && make doc
)
