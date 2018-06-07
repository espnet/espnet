#!/bin/bash

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.


if [ $# -le 3 ]; then
   echo "Arguments should be a list of WSJ directories, see ../run.sh for example."
   exit 1;
fi


dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

if [ -z $IRSTLM ] ; then
  export IRSTLM=$KALDI_ROOT/tools/irstlm/
fi
export PATH=${PATH}:$IRSTLM/bin
if ! command -v prune-lm >/dev/null 2>&1 ; then
  echo "$0: Error: the IRSTLM is not available or compiled" >&2
  echo "$0: Error: We used to install it by default, but." >&2
  echo "$0: Error: this is no longer the case." >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_irstlm.sh" >&2
  exit 1
fi

cd $dir
# Make directory of links to the WSJ disks such as 11-13.1.  This relies on the command
# line arguments being absolute pathnames.
rm -r links/ 2>/dev/null
mkdir links/
ln -s $* links

# Do some basic checks that we have what we expected.
if [ ! -d links/11-13.1 -o ! -d links/13-34.1 -o ! -d links/11-2.1 ]; then
  echo "wsj_data_prep.sh: Spot check of command line arguments failed"
  echo "Command line arguments must be absolute pathnames to WSJ directories"
  echo "with names like 11-13.1."
  echo "Note: if you have old-style WSJ distribution,"
  echo "local/cstr_wsj_data_prep.sh may work instead, see run.sh for example."
  exit 1;
fi

# This version for SI-84

cat links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl $* | sort | \
 grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_si84.flist

nl=`cat train_si84.flist | wc -l`
[ "$nl" -eq 7138 ] || echo "Warning: expected 7138 lines in train_si84.flist, got $nl"

# This version for SI-284
cat links/13-34.1/wsj1/doc/indices/si_tr_s.ndx \
 links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl  $* | sort | \
 grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_si284.flist

nl=`cat train_si284.flist | wc -l`
[ "$nl" -eq 37416 ] || echo "Warning: expected 37416 lines in train_si284.flist, got $nl"

# Now for the test sets.
# links/13-34.1/wsj1/doc/indices/readme.doc
# describes all the different test sets.
# Note: each test-set seems to come in multiple versions depending
# on different vocabulary sizes, verbalized vs. non-verbalized
# pronunciations, etc.  We use the largest vocab and non-verbalized
# pronunciations.
# The most normal one seems to be the "baseline 60k test set", which
# is h1_p0.

# Nov'92 (333 utts)
# These index files have a slightly different format;
# have to add .wv1
cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
  $local/ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > test_eval92.flist

# Nov'92 (330 utts, 5k vocab)
cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx | \
  $local/ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > test_eval92_5k.flist

# Nov'93: (213 utts)
# Have to replace a wrong disk-id.
cat links/13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx | \
  sed s/13_32_1/13_33_1/ | \
  $local/ndx2flist.pl $* | sort > test_eval93.flist

# Nov'93: (213 utts, 5k)
cat links/13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx | \
  sed s/13_32_1/13_33_1/ | \
  $local/ndx2flist.pl $* | sort > test_eval93_5k.flist

# Dev-set for Nov'93 (503 utts)
cat links/13-34.1/wsj1/doc/indices/h1_p0.ndx | \
  $local/ndx2flist.pl $* | sort > test_dev93.flist

# Dev-set for Nov'93 (513 utts, 5k vocab)
cat links/13-34.1/wsj1/doc/indices/h2_p0.ndx | \
  $local/ndx2flist.pl $* | sort > test_dev93_5k.flist


# Dev-set Hub 1,2 (503, 913 utterances)

# Note: the ???'s below match WSJ and SI_DT, or wsj and si_dt.
# Sometimes this gets copied from the CD's with upcasing, don't know
# why (could be older versions of the disks).
find `readlink links/13-16.1`/???1/??_??_20 -print | grep -i ".wv1" | sort > dev_dt_20.flist
find `readlink links/13-16.1`/???1/??_??_05 -print | grep -i ".wv1" | sort > dev_dt_05.flist


# Finding the transcript files:
for x in $*; do find -L $x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
   $local/flist2scp.pl $x.flist | sort > ${x}_sph.scp
   cat ${x}_sph.scp | awk '{print $1}' | $local/find_transcripts.pl  dot_files.flist > $x.trans1
done

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
   cat $x.trans1 | $local/normalize_transcript.pl $noiseword | sort > $x.txt || exit 1;
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in train_si84 train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
   cat ${x}_sph.scp | awk '{print $1}' | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
   cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done


#in case we want to limit lm's on most frequent words, copy lm training word frequency list
cp links/13-32.1/wsj1/doc/lng_modl/vocab/wfl_64.lst $lmdir
chmod u+w $lmdir/*.lst # had weird permissions on source.

echo "Data preparation succeeded"
