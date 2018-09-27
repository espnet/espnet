#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -o pipefail
. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="201"
seq_type="grapheme"
phn_ali=

. ./utils/parse_options.sh

cwd=$(utils/make_absolute.sh `pwd`)
echo "stage 1: Setup language specific directories"
# Create a language specific directory for each language
echo "Languages: ${langs}"
echo "Target Sequence type: $seq_type"
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

  cp ${cwd}/cmd.sh .
  cp ${cwd}/path_babel.sh path.sh
  
  cd ${cwd}
done

# Set up recog lang
[ -d data/${recog} ] || mkdir -p data/${recog}
cd data/${recog}
ln -sf ${cwd}/local .
for f in ${cwd}/{utils,steps,conf}; do
  link=`make_absolute.sh $f`
  ln -sf $link .
done

cp ${cwd}/cmd.sh .
cp ${cwd}/path_babel.sh path.sh
cd ${cwd}


# Prepare data
for l in ${langs}; do
  cd data/${l}
  #############################################################################
  # Prepare the data directories (train and dev10h) directories
  #############################################################################
  if [ $seq_type = "phoneme" ]; then
    echo "ALI: $phn_ali"
    if [ ! -z $phn_ali ]; then
      ./local/prepare_data.sh --extract-feats false ${l}
    else
      ./local/prepare_data.sh --extract-feats true ${l}
      mkdir -p data/lang
      if [[ ! -f data/lang/L.fst || data/lang/L.fst -ot data/local/lexicon.txt ]]; then
        echo ------------------------------------------------------------------
        echo "Creating L.fst etc in data/lang on" `date`
        echo ------------------------------------------------------------------
        utils/prepare_lang.sh \
          --share-silence-phones true \
          data/local $oovSymbol data/local/tmp.lang data/lang
      fi

    fi
    # Make an attempt at merging shared phonemes across different languages
    # including splitting diphthongs and triphthongs.
    ./local/prepare_universal_dict.sh --dict data/dict_universal ${l} 
  else
    ./local/prepare_data.sh ${l}
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
  if [ ! -z $phn_ali ]; then
    ./utils/prepare_lang.sh --share-silence-phones true \
      data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal
  fi
fi

cd data/${recog}
./local/prepare_recog.sh ${recog}
cd ${cwd}

