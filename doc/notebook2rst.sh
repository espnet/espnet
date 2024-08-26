#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d notebook ]; then
    git clone https://github.com/espnet/notebook --depth 1
fi

. ../tools/activate_python.sh

echo "# Notebook

Jupyter notebooks for course demos and tutorials.
"

cd notebook
for basedir in */; do
    printf '## %s\n' "$basedir"
    find ${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c 'jupyter nbconvert --clear-output "$1"' shell {} \;
    find ./${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c 'jupyter nbconvert --to markdown "$1"' shell {} \;

    while IFS= read -r -d '' md_file; do
        filename=$(basename ${md_file})
        echo "* [${filename}](./${md_file:((${#basedir})):100})"
    done <   <(find ${basedir} -name "*.md" -print0)

    while IFS= read -r -d '' ipynb_file; do
        rm ${ipynb_file}
    done <   <(find ${basedir} -name "*.ipynb" -print0)

    # generate README.md
    echo "# ${basedir} Demo" > ${basedir}README.md
    while IFS= read -r -d '' md_file; do
        filename=$(basename ${md_file})
        echo "* [${filename}](./${md_file:((${#basedir})):100})" >> ${basedir}README.md
    done <   <(find ${basedir} -name "*.md" -print0)
    echo ""
done
