#!/usr/bin/env bash

# to suppress errors during doc generation of utils/ when USE_CONDA=false in travis
mkdir -p tools/venv/bin
touch tools/venv/bin/activate
. tools/venv/bin/activate

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

# build sphinx document under doc/
mkdir -p doc/_gen

# NOTE allow unbound variable (-u) inside kaldi scripts
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
set -euo pipefail
# generate tools doc
(
    cd ./utils
    ../doc/argparse2rst.py ./*.py > ../doc/_gen/utils_py.rst
)

./doc/argparse2rst.py ./espnet/bin/*.py > ./doc/_gen/espnet_bin.rst


find ./utils/*.sh -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/utils_sh.rst

# generate package doc
./doc/module2rst.py espnet ./doc --exclude espnet.bin

# build html
travis-sphinx build --source=doc --nowarn

touch doc/build/.nojekyll

