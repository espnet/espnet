#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

###########################################################################
# Create dictionaries with split diphthongs and standardized tones
# This script recreates the dictionary directories by modifying the
# the phonemic inventory of the languages (according to local/phone_maps).
# All diphthongs and triphthongs are split into their constituent phones when
# possible, all tone markings, which have no standard representation across
# languages in the x-sampa phoneme set, are changed so as to be shared across
# languages when possible.
###########################################################################

dict=data/dict_universal

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_dictionary.sh --dict data/dict_universal <lang_id>"
  exit 1
fi 

l=$1

mkdir -p $dict

echo "Making dictionary for ${l}"

# Create silence lexicon (This is the set of non-silence phones standardly
# used in the babel recipes
echo -e "<silence>\tSIL\n<unk>\t<oov>\n<noise>\t<sss>\n<v-noise>\t<vns>" \
  > ${dict}/silence_lexicon.txt

# Create non-silence lexicon
grep -vFf ${dict}/silence_lexicon.txt data/local/lexicon.txt \
  > data/local/nonsilence_lexicon.txt

# Create split diphthong and standarized tone lexicons for nonsilence words
./local/prepare_universal_lexicon.py \
  ${dict}/nonsilence_lexicon.txt data/local/nonsilence_lexicon.txt \
  local/phone_maps/${l} 

cat ${dict}/{,non}silence_lexicon.txt | sort > ${dict}/lexicon.txt

# Prepare the rest of the dictionary directory
# -----------------------------------------------
./local/prepare_dict.py \
  --silence-lexicon ${dict}/silence_lexicon.txt ${dict}/lexicon.txt ${dict}


