#!/usr/bin/env bash

# to suppress errors during doc generation of utils/ when USE_CONDA=false in travis
mkdir -p tools/venv/bin
touch tools/venv/bin/activate
. tools/venv/bin/activate

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

(
    cd ./tools/kaldi/tools || exit 1
    echo "" > extras/check_dependencies.sh
    chmod +x extras/check_dependencies.sh
    make sph2pipe sclite
)

# download pre-built kaldi binary
[ ! -e ubuntu16-featbin.tar.gz ] && wget https://18-198329952-gh.circle-artifacts.com/0/home/circleci/repo/ubuntu16-featbin.tar.gz
tar -xf ./ubuntu16-featbin.tar.gz
cp featbin/* tools/kaldi/src/featbin/

cd ./egs/mini_an4/asr1 || exit 1
./run.sh
