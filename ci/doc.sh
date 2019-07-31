#!/usr/bin/env bash


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

(
    cd ./doc
    if [ ! -e notebook ]; then
        git clone https://github.com/espnet/notebook
        cd notebook
        git checkout de000a4fd000dcffa5c0ee14fc477f96b7af6fea
    fi
)

./doc/argparse2rst.py ./espnet/bin/*.py > ./doc/_gen/espnet_bin.rst


find ./utils/*.sh -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/utils_sh.rst

# generate package doc
./doc/module2rst.py espnet ./doc --exclude espnet.bin

# build html
travis-sphinx build --source=doc --nowarn

touch doc/build/.nojekyll

