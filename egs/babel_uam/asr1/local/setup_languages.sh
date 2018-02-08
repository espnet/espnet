#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -o pipefail
. ./path.sh
. ./cmd.sh
. ./conf/common_vars.sh

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306
       401 402 403"
seq_type="grapheme"

. ./utils/parse_options.sh

cwd=$(utils/make_absolute.sh `pwd`)
echo "stage 0: Setup language specific directories"
# Create a language specific directory for each language
for l in ${langs}; do
  [ -d data/${l} ] || mkdir -p data/${l}
  cd data/${l}

  # Copy the main directories from the top level into
  # each language specific directory
  ln -sf ${cwd}/local .
  for f in ${cwd}/{utils,steps,conf}; do
    link=`make_absolute.sh $f`
    ln -sf $link .
  done

  conf_file=`find conf/lang -name "${l}-*limitedLP*.conf" \
                         -o -name "${l}-*LLP*.conf" | head -1`

  echo "----------------------------------------------------"
  echo "Using language configurations: ${conf_file}"
  echo "----------------------------------------------------"

  cp ${conf_file} lang.conf
  cp ${cwd}/cmd.sh .
  cp ${cwd}/path_babel.sh path.sh
  cd ${cwd}
done

for l in ${langs}; do
  cd data/${l}
  #############################################################################
  # Prepare the data directories (train and dev10h) directories
  #############################################################################
  if [ $seq_type = "phoneme" ]; then
    ./local/prepare_data.sh --extract-feats true
    ./local/prepare_universal_dict.sh --dict data/dict_universal ${l} 
  else
    ./local/prepare_data.sh
  fi
  cd ${cwd}
done

###############################################################################
# Combine all langauge specific training directories and generate a single
# lang directory by combining all langauge specific dictionaries
###############################################################################

train_dirs=""
dict_dirs=""
for l in ${langs}; do
  train_dirs="data/${l}/data/train_${l} ${train_dirs}"
  if [ $seq_type = "phoneme" ]; then
    dict_dirs="data/${l}/data/dict_universal ${dict_dirs}"
  fi
done

./utils/combine_data.sh data/train $train_dirs

if [ $seq_type = "phoneme" ]; then
  ./local/combine_lexicons.sh data/dict_universal $dict_dirs
  # Prepare lang directory
  ./utils/prepare_lang.sh --share-silence-phones true \
    data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal
fi

