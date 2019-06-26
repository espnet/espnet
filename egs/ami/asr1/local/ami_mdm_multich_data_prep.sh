#!/bin/bash

# Copyright 2019  Johns Hopkins University (Author: Tianzi Wang)
# AMI Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

. ./path.sh

#check existing directories
if [ $# != 2 ]; then
  echo "Usage: $0 </path/to/AMI-MDM> <mic-id>"
  echo "e.g. $0 /foo/bar/AMI mdm8"
  exit 1;
fi

AMI_DIR=$1
mic=$2

SEGS=data/local/annotations/train.txt
dir=data/local/$mic/train
odir=data/$mic/train_orig
mkdir -p $dir

# Audio data directory check
if [ ! -d $AMI_DIR ]; then
  echo "Error: run.sh requires a directory argument"
  exit 1;
fi

# And transcripts check
if [ ! -f $SEGS ]; then
  echo "Error: File $SEGS no found (run ami_text_prep.sh)."
  exit 1;
fi

#find multichannel mics
find $AMI_DIR -iname "*.Array1-0[0-8].wav" | sort | sed "/TS3007c/d" > $dir/wav.flist

n=`cat $dir/wav.flist | wc -l`
echo "In total, $n multiple distant mic files were found."
[ $n -ne 1344 ] && \
  echo Warning: expected 1344 data data files, found $n

# (1a) Transcriptions preparation
# here we start with rt09 transcriptions, hence not much to do

awk '{meeting=$1; channel="MDM"; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq | sed "/TS3007c/d" > $dir/text 

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " "
}' < $dir/text > $dir/segments

# Fix text because local/*_noisy_chime4_data_prep.sh assumes 1ch_track originally.
for ch in 1 2 3 4 5 6 7 8; do    
    # e.g. AMI_EN2001a_MDM -> AMI_EN2001a_MDM.Array1-01
    <$dir/text sed -r "s/(AMI_\w*_MDM_\w*) /\1.Array1-0${ch} /g"
done | sort > $dir/text.tmp

for ch in 1 2 3 4 5 6 7 8; do    
    # e.g. AMI_EN2001a_MDM -> AMI_EN2001a_MDM.Array1-01
    <$dir/segments sed -r "s/(AMI_\w*_MDM_\w*) (AMI_\w*_MDM) /\1.Array1-0${ch} \2.Array1-0${ch} /g"
done | sort -k 2 > $dir/segments.tmp

#EN2001a.Array1-01.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $dir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\.(.*)/; print "AMI_$1_MDM.$2\n"' | \
  paste - $dir/wav.flist > $dir/wav1.scp

#Keep only training part of waves
awk '{print $2}' $dir/segments.tmp | sort -u | join - $dir/wav1.scp | sort > $dir/wav2.scp

#Two distant recordings are missing, agree segments with wav.scp
awk '{print $1}' $dir/wav2.scp | join -2 2 - $dir/segments.tmp | \
    awk '{print $2" "$1" "$3" "$4" "$5}' | sort > $dir/s; mv $dir/s $dir/segments

#...and text with segments
awk '{print $1}' $dir/segments | join - $dir/text.tmp > $dir/t; mv $dir/t $dir/text

# fix multichannel format
mkdir -p $dir/multich

#replace path with an appropriate sox command that mix multi channel
for ch in 1 2 3 4 5 6 7 8; do
    <$dir/wav2.scp grep "Array1-0${ch}" | sed -r 's/^(.*?).Array1-0[0-9] /\1 /g' > $dir/wav_ch${ch}.scp
done
mix-mono-wav-scp.py $dir/wav_ch*.scp | sed -r 's/-t wav /-t wavpcm -e signed-integer '/g> $dir/multich/wav.scp
rm -f $dir/wav_ch*.scp

#awk '{print $1" sox -c 1 -t wavpcm -e signed-integer "$2" -t wavpcm - |"}' $dir/wav2.scp > $dir/wav.scp

#prep reco2file_and_channel
cat $dir/multich/wav.scp | \
  perl -ane '$_ =~ m:^(\S+MDM).*\/([IETB].*)\.Array1-0[0-9].*$: || die "bad label $_";
       print "$1 $2 A\n"; ' | sort -u > $dir/multich/reco2file_and_channel || exit 1;

<$dir/segments sed -r 's/\.Array1-0[0-9]//g' | sort -u >$dir/multich/segments
<$dir/text sed -r 's/\.Array1-0[0-9]//g' | sort -u > $dir/multich/text

# In this data-prep phase we adapt to the session only [later on we may split
# into shorter pieces].
# We use the first two underscore-separated fields of the utterance-id
# as the speaker-id, e.g. 'AMI_EN2001a_MDM_FEO065_0090130_0090775' becomes 'AMI_EN2001a'.
awk '{print $1}' $dir/multich/segments | \
  perl -ane 'chop; @A = split("_", $_); $spkid = join("_", @A[0,1]); print "$_ $spkid\n";'  \
  >$dir/multich/utt2spk || exit 1;

utils/utt2spk_to_spk2utt.pl <$dir/multich/utt2spk >$dir/multich/spk2utt || exit 1;

# Copy stuff into its final locations
mkdir -p $odir
for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp $dir/multich/$f $odir/$f | exit 1;
done

rm $dir/*.tmp

utils/validate_data_dir.sh --no-feats $odir

echo AMI MDM multichannel data preparation succeeded.

