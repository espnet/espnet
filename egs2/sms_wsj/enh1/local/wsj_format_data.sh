#!/usr/bin/env bash

# Copyright 2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
#           2015  Guoguo Chen
# Apache 2.0

# Modified from the original version in Kaldi

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/lang_test_ug,
# data/train_si284, data/train_si84, etc.

# Don't bother doing train_si84 separately (although we have the file lists
# in data/local/) because it's just the first 7138 utterances in train_si284.
# We'll create train_si84 after doing the feature extraction.

data_dir=data

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/local/data

for x in train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
  mkdir -p ${data_dir}/$x
  cp $srcdir/${x}_wav.scp ${data_dir}/$x/wav.scp || exit 1;
  cp $srcdir/$x.txt ${data_dir}/$x/text || exit 1;
  cp $srcdir/$x.spk2utt ${data_dir}/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk ${data_dir}/$x/utt2spk || exit 1;
  utils/filter_scp.pl ${data_dir}/$x/spk2utt $srcdir/spk2gender > ${data_dir}/$x/spk2gender || exit 1;
done

echo "Succeeded in formatting data."
