#!/usr/bin/env bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"
FLP=true

. ./utils/parse_options.sh

set -e
set -o pipefail

all_langs=""
for l in `cat <(echo ${langs}) <(echo ${recog}) | tr " " "\n" | sort -u`; do
  all_langs="${l} ${all_langs}"
done
all_langs=${all_langs%% }

# Save top-level directory
cwd=$(utils/make_absolute.sh `pwd`)
echo "Stage 0: Setup Language Specific Directories"

echo " --------------------------------------------"
echo "Languagues: ${all_langs}"

# Basic directory prep
for l in ${all_langs}; do
  [ -d data/${l} ] || mkdir -p data/${l}
  cd data/${l}

  ln -sf ${cwd}/local .
  for f in ${cwd}/{utils,steps,conf}; do
    link=`make_absolute.sh $f`
    ln -sf $link .
  done

  cp ${cwd}/cmd.sh .
  cp ${cwd}/path.sh .
  sed -i 's/\.\.\/\.\.\/\.\./\.\.\/\.\.\/\.\.\/\.\.\/\.\./g' path.sh

  cd ${cwd}
done

# Prepare language specific data
for l in ${all_langs}; do
  (
    cd data/${l}
    ./local/prepare_data.sh --FLP ${FLP} ${l}
    cd ${cwd}
  ) &
done
wait

# Combine all language specific training directories and generate a single
# lang directory by combining all language specific dictionaries
train_dirs=""
dev_dirs=""
eval_dirs=""
for l in ${langs}; do
  train_dirs="data/${l}/data/train_${l} ${train_dirs}"
done

for l in ${recog}; do
  dev_dirs="data/${l}/data/dev_${l} ${dev_dirs}"
done

./utils/combine_data.sh data/train ${train_dirs}
./utils/combine_data.sh data/dev ${dev_dirs}

for l in ${recog}; do
  ln -s ${cwd}/data/${l}/data/eval_${l} ${cwd}/data/eval_${l}
done
