#!/usr/bin/env bash

# build sphinx document under doc/
mkdir -p doc/_gen
./doc/gen_rst_py_tool.py ./espnet/bin/*.py > ./doc/_gen/espnet_bin.rst
# ./utils/gen_rst_py_bin.py ./utils/*.py > ./doc/_gen/utils_py.rst
# ./utils/gen_rst_sh_bin.py ./utils/*.sh > ./doc/_gen/utils_sh.rst


travis-sphinx build --source=doc --nowarn
