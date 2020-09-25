#!/usr/bin/env bash

set -euo pipefail

# install kaldi
[ ! -d tools/kaldi ] && git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
(
    cd ./tools/kaldi/tools || exit 1
    echo "" > extras/check_dependencies.sh
    chmod +x extras/check_dependencies.sh
)

# download pre-built kaldi binary
# TODO(karita) support non ubuntu env
[ ! -e ubuntu16-featbin.tar.gz ] && wget --tries=3 https://github.com/espnet/kaldi-bin/releases/download/v0.0.1/ubuntu16-featbin.tar.gz
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* tools/kaldi/src/featbin/
