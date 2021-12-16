#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi


if [ ! -e sph2pipe_v2.5.tar.gz ]; then
    wget -T 10 https://github.com/espnet/kaldi-bin/releases/download/v0.0.2/sph2pipe_v2.5.tar.gz || \
	wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz || \
	wget --no-check-certificate -T 10  https://sourceforge.net/projects/kaldi/files/sph2pipe_v2.5.tar.gz
fi

if [ ! -e sph2pipe_v2.5 ]; then
    tar --no-same-owner -xzf sph2pipe_v2.5.tar.gz
fi

(
    set -euo pipefail
	cd sph2pipe_v2.5
	gcc -o sph2pipe  *.c -lm
)
