#!/usr/bin/env bash

set -euo pipefail

# install kaldi
[ ! -d tools/kaldi ] && git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
(
    cd ./tools/kaldi/tools || exit 1
    echo "" > extras/check_dependencies.sh
    chmod +x extras/check_dependencies.sh
    make sph2pipe sclite
)

# download pre-built kaldi binary
# TODO(karita) support non ubuntu env
[ ! -e ubuntu16-featbin.tar.gz ] && wget https://18-198329952-gh.circle-artifacts.com/0/home/circleci/repo/ubuntu16-featbin.tar.gz
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* tools/kaldi/src/featbin/
