#!/bin/bash

set -e
set -u
set -o pipefail

export ESPNET_PYTHON_VERSION=3.10
export TH_VERSION=1.13.1
export CHAINER_VERSION=6.0.0


conda install -y python=${ESPNET_PYTHON_VERSION}

rm -rf ./egs2/mini_an4/**/data
# rm -rf ./egs2/mini_an4/**/downloads
rm -rf ./egs2/mini_an4/**/dump
rm -rf ./egs2/mini_an4/**/exp

rm -rf shellcheck-*
rm -rf ubuntu16-*
rm -rf featbin
MAIN=${PWD}

cd ./tools

make clean python
make clean

cd ${MAIN}

./ci/install.sh
# ./ci/test_shell_espnet2.sh
# ./ci/test_python_espnet2.sh

# ./ci/install_kaldi.sh
# ./ci/test_integration_espnet2.sh
