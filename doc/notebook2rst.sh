#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d notebook ]; then
    git clone https://github.com/espnet/notebook --depth 1
fi

echo "# Notebook

Jupyter notebooks for course demos and tutorials.
"

cd notebook
for basedir in */; do
    printf "## ${basedir}\n"
    find ${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c ". ../../tools/activate_python.sh;jupyter nbconvert --clear-output \"{}\"" \;
    find ./${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c ". ../../tools/activate_python.sh;jupyter nbconvert --to markdown \"{}\"" \;

    for md_file in `find ${basedir} -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:((${#basedir})):100})"
    done
    for ipynb_file in `find ${basedir} -name "*.ipynb"`; do
        rm ${ipynb_file}
    done

    # generate README.md
    echo "# ${basedir} Demo" > ${basedir}README.md
    for md_file in `find ${basedir} -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:((${#basedir})):100})" >> ${basedir}README.md
    done
    echo ""
done
