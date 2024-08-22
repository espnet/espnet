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
    printf "## %s\n" "${basedir}"
    find ${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c '. ../../tools/activate_python.sh;jupyter nbconvert --clear-output "$1"' shell {} \;
    find ./${basedir} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c '. ../../tools/activate_python.sh;jupyter nbconvert --to markdown "$1"' shell {} \;

    # generate README.md
    echo "# ${basedir} Demo" > ${basedir}README.md
    find "$basedir" -type f -name "*.md" | while read -r md_file; do
        filename=`basename ${md_file}`
        line="- [${filename}](.${md_file:((${#basedir})):100})"
        echo $line
        echo $line >> ${basedir}README.md
    done
    echo ""
done
