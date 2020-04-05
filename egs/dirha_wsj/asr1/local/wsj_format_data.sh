#!/bin/bash

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
#           2015  Guoguo Chen
# Apache 2.0

# Modified from the script for dirha_wsj
# Xiaofei Wang 06/01/2019

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

lang_suffix=
mic=$1

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/local/data

for x in train_si284; do
  mkdir -p data/${x}_$mic
  cp $srcdir/${x}_wav.scp data/${x}_$mic/wav.scp || exit 1;
  cp $srcdir/$x.txt data/${x}_$mic/text || exit 1;
  cp $srcdir/$x.spk2utt data/${x}_$mic/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/${x}_$mic/utt2spk || exit 1;
  utils/filter_scp.pl data/${x}_$mic/spk2utt $srcdir/spk2gender > data/${x}_$mic/spk2gender || exit 1;
done

echo "Succeeded in formatting data."
