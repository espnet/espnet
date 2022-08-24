#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d notebook ]; then
    git clone https://github.com/espnet/notebook --depth 1
fi

echo "\
.. toctree::
   :maxdepth: 1
   :caption: Notebook:
"

find ./notebook/*.ipynb -exec echo "   {}" \;
