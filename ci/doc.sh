#!/usr/bin/env bash

. tools/activate_python.sh

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
# FIXME
# ./doc/argparse2rst.py ./espnet2/bin/*.py > ./doc/_gen/espnet2_bin.rst


find ./utils/*.sh tools/sentencepiece_commands/spm_* -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/utils_sh.rst
find ./espnet2/bin/*.py -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/espnet2_bin.rst

./doc/notebook2rst.sh > ./doc/_gen/notebooks.rst

# generate package doc
./doc/module2rst.py --root espnet espnet2 --dst ./doc --exclude espnet.bin

# build html
# TODO(karita): add -W to turn warnings into errors
sphinx-build -b html doc doc/build

touch doc/build/.nojekyll
