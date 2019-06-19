#!/usr/bin/env bash

set -euo pipefail

# build sphinx document under doc/
mkdir -p doc/_gen

./doc/gen_rst_py_tool.py ./espnet/bin/*.py > ./doc/_gen/espnet_bin.rst

(
    cd ./utils
    ../doc/gen_rst_py_tool.py ./*.py > ../doc/_gen/utils_py.rst
)

# ./utils/gen_rst_sh_tool.py ./utils/*.sh > ./doc/_gen/utils_sh.rst

travis-sphinx build --source=doc --nowarn
