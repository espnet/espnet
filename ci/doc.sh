#!/usr/bin/env bash

. tools/activate_python.sh

build_and_convert () {
    # $1: path
    # $2: output
    mkdir -p ./doc/_gen/tools/$2
    for filename in `find $1`; do
        bn=`basename ${filename}`
        echo "Converting ${filename} to rst..."
        ./doc/usage2rst.sh ${filename} > ./doc/_gen/tools/$2/${bn}.rst
    done
}

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

# build sphinx document under doc/
mkdir -p doc/_gen
mkdir -p doc/_gen/tools
mkdir -p doc/_gen/guide

# NOTE allow unbound variable (-u) inside kaldi scripts
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
set -euo pipefail
# generate tools doc
mkdir utils_py
./doc/argparse2rst.py \
    --title utils_py \
    --output_dir utils_py \
    ./utils/*.py 
mv utils_py ./doc/_gen/tools

mkdir espnet_bin
./doc/argparse2rst.py \
    --title espnet_bin \
    --output_dir espnet_bin \
    ./espnet/bin/*.py 
mv espnet_bin ./doc/_gen/tools

mkdir espnet2_bin
./doc/argparse2rst.py \
    --title espnet2_bin \
    --output_dir espnet2_bin \
    ./espnet2/bin/*.py 
mv espnet2_bin ./doc/_gen/tools

build_and_convert "utils/*.sh" utils
build_and_convert "tools/sentencepiece_commands/spm_*" spm

./doc/notebook2rst.sh > ./doc/notebooks.md

# generate package doc
python ./doc/members2rst.py --root espnet --dst ./doc/_gen/guide --exclude espnet.bin
python ./doc/members2rst.py --root espnet2 --dst ./doc/_gen/guide --exclude espnet2.bin
python ./doc/members2rst.py --root espnetez --dst ./doc/_gen/guide

# generate package doc
./doc/module2rst.py --root espnet espnet2 --dst ./doc --exclude espnet.bin

# build markdown
cp ./doc/index.rst ./doc/_gen/index.rst
cp ./doc/conf.py ./doc/_gen/
rm -f ./doc/_gen/tools/espnet2_bin/*_train.rst
sphinx-build -M markdown ./doc/_gen ./doc/build

# copy markdown files to specific directory.
cp -r ./doc/build/markdown/* ./doc/vuepress/src/
cp -r ./doc/notebook ./doc/vuepress/src/
cp ./doc/*.md ./doc/vuepress/src/
mv ./doc/vuepress/src/README.md ./doc/vuepress/src/document.md
cp -r ./doc/image ./doc/vuepress/src/

# Document generation has finished.
# From the following point we modify files for VuePress.
# Replace language tags to supported language tags
find ./doc/vuepress/src/ -name "*.md" -exec sed -i 's/```default/```text/g' {} \;
find ./doc/vuepress/src/ -name "*.md" -exec sed -i 's/```pycon/```python/g' {} \;
find ./doc/vuepress/src/ -name "*.md" -exec sed -i 's/```cd/```text/g' {} \;

# And convert custom tags to &lt; and &gt;, as <custom tag> can be recognized a html tag.
python ./doc/convert_custom_tags_to_html.py ./doc/vuepress/src/

# Convert API document to specific html tags to display sphinx style
python ./doc/convert_md_to_homepage.py ./doc/vuepress/src/guide/
python ./doc/convert_md_to_homepage.py ./doc/vuepress/src/tools/

# Create navbar and sidebar.
cd ./doc/vuepress
python create_menu.py --root ./src

# check if node is installed
if which node > /dev/null
then
    echo "node is installed, skipping..."
else
    apt install -y nodejs npm
    npm install n -g
    n stable
    apt purge -y nodejs npm
    apt autoremove -y
fi

npm i
# npm run docs:dev
npm run docs:build
mv src/.vuepress/dist ../../

touch doc/build/.nojekyll
