#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.

. ./conf/lang.conf
. ./path.sh
. ./cmd.sh

extract_feats=false

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_data.sh [opts] <lang_id>"
  echo >&2 "       --extract-feats :  Extract plp features for train directory"
  exit 1
fi

l=$1

#set -e           #Exit on non-zero return code from any command
#set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
#set -u           #Fail on an undefined variable

lexicon=data/local/lexicon.txt

./local/check_tools.sh || exit 1

#Preparing train directories
if [ ! -f data/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------
    train_data_dir=train_data_dir_${l}
    train_data_list=train_data_list_${l}
    local/make_corpus_subset.sh "${!train_data_dir}" "${!train_data_list}" ./data/raw_train_data
    train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`
    touch data/raw_train_data/.done
fi
nj_max=`cat ${!train_data_list} | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`

mkdir -p data/local
lexicon_file=lexicon_file_${l}
lexiconFlags=lexiconFlags_${l}
if [[ ! -f $lexicon || $lexicon -ot "${!lexicon_file}" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------
  local/make_lexicon_subset.sh $train_data_dir/transcription ${!lexicon_file} data/local/filtered_lexicon.txt
  local/prepare_lexicon.pl ${!lexiconFlags} data/local/filtered_lexicon.txt data/local
fi


if [[ ! -f data/train/wav.scp || data/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic training lists in data/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/train
  local/prepare_acoustic_training_data.pl \
    --vocab $lexicon --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log
fi

if $extract_feats; then
  echo ---------------------------------------------------------------------
  echo "Starting plp feature extraction for data/train in plp on" `date`
  echo ---------------------------------------------------------------------
  steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp_pitch/train plp
  utils/fix_data_dir.sh data/train
  steps/compute_cmvn_stats.sh data/train exp/make_plp_pitch/train plp
  utils/fix_data_dir.sh data/train
  touch data/train/.plp.done
fi


###########################################################################
# Prepend language ID to all utterances to disambiguate between speakers
# of different languages sharing the same speaker id.
#
# The individual lang directories can be used for alignments, while a
# combined directory will be used for training. This probably has minimal
# impact on performance as only words repeated across languages will pose
# problems and even amongst these, the main concern is the <hes> marker.
###########################################################################
echo "Prepend ${l} to data dir"
./utils/copy_data_dir.sh --spk-prefix "${l}_" --utt-prefix "${l}_" \
  data/train data/train_${l}
