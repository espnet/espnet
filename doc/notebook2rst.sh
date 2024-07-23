#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d notebook ]; then
    git clone https://github.com/espnet/notebook --depth 1
fi

echo "#  Notebook

Jupyter notebooks for course demos and tutorials.
"

cd notebook
documents=`for i in $(ls -d */); do echo ${i%%/}; done`
for document in "${documents[@]}"; do
    echo "## ${document}\n"
    find ./${document} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c ". ../../tools/activate_python.sh;jupyter nbconvert --clear-output \"{}\"" \;
    find ./${document} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c ". ../../tools/activate_python.sh;jupyter nbconvert --to markdown \"{}\"" \;
    
    basedir=./${document}
    for md_file in `find "./${document}" -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:((${#basedir}+1)):100})"
    done
    for ipynb_file in `find "./${document}" -name "*.ipynb"`; do
        rm ${ipynb_file}
    done

    # generate README.md
    echo "# ${document} Demo" > ./${document}/README.md
    for md_file in `find "./${document}" -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:((${#basedir}+1)):100})" >> ./${document}/README.md
    done
    echo ""
done
