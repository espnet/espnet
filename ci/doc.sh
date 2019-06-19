#!/usr/bin/env bash

set -euo pipefail

# build sphinx document under doc/
mkdir -p doc/_gen

./doc/argparse2rst.py ./espnet/bin/*.py > ./doc/_gen/espnet_bin.rst

(
    cd ./utils
    ../doc/argparse2rst.py ./*.py > ../doc/_gen/utils_py.rst
)

find ./utils/*.sh -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/utils_sh.rst

travis-sphinx build --source=doc --nowarn
