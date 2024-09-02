#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d notebook ]; then
    git clone https://github.com/espnet/notebook --depth 1
fi

. ../tools/activate_python.sh

cd notebook

# ipynb -> md
for basedir in */; do
    find ${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c 'jupyter nbconvert --clear-output "$1"' shell {} \;
    find ./${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c 'jupyter nbconvert --to markdown "$1"' shell {} \;
done

# Update README.md
# 1. Add table of contents
sed -i '1 a [[toc]]' README.md
# 2. Change link inside the original README.md
sed -i "s/\.ipynb)/\.md)/g" README.md

