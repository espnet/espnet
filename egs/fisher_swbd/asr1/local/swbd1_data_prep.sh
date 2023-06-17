#!/usr/bin/env bash

# Switchboard-1 training data preparation customized for Edinburgh
# Author:  Arnab Ghoshal (Jan 2013)

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

. ./path.sh

#check existing directories
if [ $# != 1 ]; then
  echo "Usage: swbd1_data_prep.sh /path/to/SWBD"
  exit 1;
fi

SWBD_DIR=$1

dir=data/local/train_swbd
mkdir -p $dir

# Audio data directory check
if [ ! -d $SWBD_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1;
fi

sph2pipe=sph2pipe
! command -v "${sph2pipe}" &> /dev/null \
  && echo "Could not execute the sph2pipe program at $sph2pipe" && exit 1;

# Option A: SWBD dictionary file check
[ ! -f $dir/swb_ms98_transcriptions/sw-ms98-dict.text ] && \
  echo  "SWBD dictionary file does not exist" &&  exit 1;

# find sph audio files
find $SWBD_DIR -iname '*.sph' | sort > $dir/sph.flist

n=`cat $dir/sph.flist | wc -l`
[ $n -ne 2435 ] && \
  echo Warning: expected 2435 data data files, found $n


# (1a) Transcriptions preparation
# make basic transcription file (add segments info)
# **NOTE: In the default Kaldi recipe, everything is made uppercase, while we
# make everything lowercase here. This is because we will be using SRILM which
# can optionally make everything lowercase (but not uppercase) when mapping
# LM vocabs.
awk '{
       name=substr($1,1,6); gsub("^sw","sw0",name); side=substr($1,7,1);
       stime=$2; etime=$3;
       printf("%s-%s_%06.0f-%06.0f",
              name, side, int(100*stime+0.5), int(100*etime+0.5));
       for(i=4;i<=NF;i++) printf(" %s", $i); printf "\n"
}' $dir/swb_ms98_transcriptions/*/*/*-trans.text  > $dir/transcripts1.txt

# test if trans. file is sorted
export LC_ALL=C;
sort -c $dir/transcripts1.txt || exit 1; # check it's sorted.

# Remove SILENCE, <B_ASIDE> and <E_ASIDE>.

# Note: we have [NOISE], [VOCALIZED-NOISE], [LAUGHTER], [SILENCE].
# removing [SILENCE], and the <B_ASIDE> and <E_ASIDE> markers that mark
# speech to somone; we will give phones to the other three (NSN, SPN, LAU).
# There will also be a silence phone, SIL.
# **NOTE: modified the pattern matches to make them case insensitive
cat $dir/transcripts1.txt \
  | perl -ane 's:\s\[SILENCE\](\s|$):$1:gi;
               s/<B_ASIDE>//gi;
               s/<E_ASIDE>//gi;
               print;' \
  | awk '{if(NF > 1) { print; } } ' > $dir/transcripts2.txt


# **NOTE: swbd1_map_words.pl has been modified to make the pattern matches
# case insensitive
local/swbd1_map_words.pl -f 2- $dir/transcripts2.txt  > $dir/text  # final transcripts

# format acronyms in text
python3 local/map_acronyms_transcripts.py -i $dir/text -o $dir/text_map \
  -M data/local/dict_nosp/acronyms_swbd.map
cp $dir/text $dir/text_bk
mv $dir/text_map $dir/text



# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#sw02001-A_000098-001156 sw02001-A 0.98 11.56
awk '{
       segment=$1;
       split(segment,S,"[_-]");
       side=S[2]; audioname=S[1]; startf=S[3]; endf=S[4];
       print segment " " audioname "-" side " " startf/100 " " endf/100
}' < $dir/text > $dir/segments

sed -e 's?.*/??' -e 's?.sph??' $dir/sph.flist | paste - $dir/sph.flist \
  > $dir/sph.scp

awk -v sph2pipe=$sph2pipe '{
  printf("%s-A %s -f wav -p -c 1 %s |\n", $1, sph2pipe, $2);
  printf("%s-B %s -f wav -p -c 2 %s |\n", $1, sph2pipe, $2);
}' < $dir/sph.scp | sort > $dir/wav.scp || exit 1;
#side A - channel 1, side B - channel 2

# this file reco2file_and_channel maps recording-id (e.g. sw02001-A)
# to the file name sw02001 and the A, e.g.
# sw02001-A  sw02001 A
# In this case it's trivial, but in other corpora the information might
# be less obvious.  Later it will be needed for ctm scoring.
awk '{print $1}' $dir/wav.scp \
  | perl -ane '$_ =~ m:^(\S+)-([AB])$: || die "bad label $_";
               print "$1-$2 $1 $2\n"; ' \
  > $dir/reco2file_and_channel || exit 1;

awk '{spk=substr($1,1,9); print $1 " " spk}' $dir/segments > $dir/utt2spk \
  || exit 1;
sort -k 2 $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt || exit 1;

# We assume each conversation side is a separate speaker. This is a very
# reasonable assumption for Switchboard. The actual speaker info file is at:
# http://www.ldc.upenn.edu/Catalog/desc/addenda/swb-multi-annot.summary

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p data/train_swbd
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/$f data/train_swbd/$f || exit 1;
done


echo Switchboard-1 data preparation succeeded.

