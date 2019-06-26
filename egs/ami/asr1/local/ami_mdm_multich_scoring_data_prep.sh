#!/bin/bash

# Copyright 2019  Johns Hopkins University (Author: Tianzi Wang)
# AMI Corpus training data preparation
# Apache 2.0

# Note: this is called by ../run.sh.

. ./path.sh

#check existing directories
if [ $# != 3 ]; then
  echo "Usage: $0 /path/to/AMI-MDM mic-name set-name"
  echo "e.g: $0 /foo/bar/AMI mdm8 dev"
  exit 1;
fi

AMI_DIR=$1
mic=$2
SET=$3

SEGS=data/local/annotations/$SET.txt
tmpdir=data/local/$mic/$SET
dir=data/$mic/${SET}_orig

mkdir -p $tmpdir

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

# find selected mdm wav audio files only
find $AMI_DIR -iname "*.Array1-0[0-8].wav" | sort > $tmpdir/wav.flist
n=`cat $tmpdir/wav.flist | wc -l`
if [ $n -ne 1352 ]; then
  echo "Warning. Expected to find 1352 files but found $n."
fi

# (1a) Transcriptions preparation
awk '{meeting=$1; channel="MDM"; speaker=$3; stime=$4; etime=$5;
 printf("AMI_%s_%s_%s_%07.0f_%07.0f", meeting, channel, speaker, int(100*stime+0.5), int(100*etime+0.5));
 for(i=6;i<=NF;i++) printf(" %s", $i); printf "\n"}' $SEGS | sort | uniq > $tmpdir/text

# (1c) Make segment files from transcript
#segments file format is: utt-id side-id start-time end-time, e.g.:
#AMI_ES2011a_H00_FEE041_0003415_0003484
awk '{
       segment=$1;
       split(segment,S,"[_]");
       audioname=S[1]"_"S[2]"_"S[3]; startf=S[5]; endf=S[6];
       print segment " " audioname " " startf/100 " " endf/100 " "
}' < $tmpdir/text > $tmpdir/segments

# Fix text because local/*_noisy_chime4_data_prep.sh assumes 1ch_track originally.
for ch in 1 2 3 4 5 6 7 8; do
    # e.g. AMI_EN2001a_MDM -> AMI_EN2001a_MDM.Array1-01
    <$tmpdir/text sed -r "s/(AMI_\w*_MDM_\w*) /\1.Array1-0${ch} /g"
done | sort > $tmpdir/text.tmp

for ch in 1 2 3 4 5 6 7 8; do
    # e.g. AMI_EN2001a_MDM -> AMI_EN2001a_MDM.Array1-01
    <$tmpdir/segments sed -r "s/(AMI_\w*_MDM_\w*) (AMI_\w*_MDM) /\1.Array1-0${ch} \2.Array1-0${ch} /g"
done | sort -k 2 > $tmpdir/segments.tmp

#EN2001a.Array1-01.wav
#sed -e 's?.*/??' -e 's?.sph??' $dir/wav.flist | paste - $dir/wav.flist \
#  > $dir/wav.scp

sed -e 's?.*/??' -e 's?.wav??' $tmpdir/wav.flist | \
 perl -ne 'split; $_ =~ m/(.*)\.(.*)/; print "AMI_$1_MDM.$2\n"' | \
  paste - $tmpdir/wav.flist > $tmpdir/wav1.scp

#Keep only devset part of waves
awk '{print $2}' $tmpdir/segments.tmp | sort -u | join - $tmpdir/wav1.scp >  $tmpdir/wav2.scp

#Agree segments with wav.scp and deprecate missing recodings
awk '{print $1}' $tmpdir/wav2.scp | join -2 2 - $tmpdir/segments.tmp | \
    awk '{print $2" "$1" "$3" "$4" "$5}' | sort > $tmpdir/s; mv $tmpdir/s $tmpdir/segments

#...and text with segments
awk '{print $1}' $tmpdir/segments | join - $tmpdir/text.tmp > $tmpdir/t; mv $tmpdir/t $tmpdir/text
# fix multichannel format
mkdir -p $tmpdir/multich

#replace path with an appropriate sox command that mix multi channel
for ch in 1 2 3 4 5 6 7 8; do
    <$tmpdir/wav2.scp grep "Array1-0${ch}" | sed -r 's/^(.*?).Array1-0[0-9] /\1 /g' > $tmpdir/wav_ch${ch}.scp
done
mix-mono-wav-scp.py $tmpdir/wav_ch*.scp | sed -r 's/-t wav /-t wavpcm -e signed-integer '/g> $tmpdir/multich/wav.scp
rm -f $tmpdir/wav_ch*.scp

#prep reco2file_and_channel
cat $tmpdir/multich/wav.scp | \
  perl -ane '$_ =~ m:^(\S+MDM)\s+.*\/([IETB].*)\.Array1-0[0-9].*$: || die "bad label $_";
       print "$1 $2 A\n"; ' > $tmpdir/multich/reco2file_and_channel || exit 1;

<$tmpdir/segments sed -r 's/\.Array1-0[0-9]//g' | sort -u >$tmpdir/multich/segments
<$tmpdir/text sed -r 's/\.Array1-0[0-9]//g' | sort -u >$tmpdir/multich/text

# now assume we adapt to the session only
awk '{print $1}' $tmpdir/multich/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_";
          print "$1$2$3 $1\n";'  \
    > $tmpdir/multich/utt2spk || exit 1;

sort -k 2 $tmpdir/multich/utt2spk | utils/utt2spk_to_spk2utt.pl > $tmpdir/multich/spk2utt || exit 1;

# but we want to properly score the overlapped segments, hence we generate the extra
# utt2spk_stm file containing speakers ids used to generate the stms for mdm/sdm case
awk '{print $1}' $tmpdir/multich/segments | \
  perl -ane '$_ =~ m:^(\S+)([FM][A-Z]{0,2}[0-9]{3}[A-Z]*)(\S+)$: || die "bad label $_";
          print "$1$2$3 $1$2\n";' > $tmpdir/multich/utt2spk_stm || exit 1;

#check and correct case when segment timings for a given speaker overlap themself
#(important for simulatenous asclite scoring to proceed).
#There is actually only one such case for devset and automatic segmentetions
join $tmpdir/multich/utt2spk_stm $tmpdir/multich/segments | \
  awk '{ utt=$1; spk=$2; wav=$3; t_beg=$4; t_end=$5;
         if(spk_prev == spk && t_end_prev > t_beg) {
           print utt, wav, t_beg, t_end">"utt, wav, t_end_prev, t_end;
         }
         spk_prev=spk; t_end_prev=t_end;
       }' > $tmpdir/multich/segments_to_fix

if [ `cat $tmpdir/multich/segments_to_fix | wc -l` -gt 0 ]; then
  echo "$0. Applying following fixes to segments"
  cat $tmpdir/multich/segments_to_fix
  while read line; do
     p1=`echo $line | awk -F'>' '{print $1}'`
     p2=`echo $line | awk -F'>' '{print $2}'`
     sed -ir "s:$p1:$p2:" $tmpdir/multich/segments
  done < $tmpdir/multich/segments_to_fix
fi

# Copy stuff into its final locations [this has been moved from the format_data
# script]
mkdir -p $dir
for f in spk2utt utt2spk utt2spk_stm wav.scp text segments reco2file_and_channel; do
  cp $tmpdir/multich/$f $dir/$f || exit 1;
done

rm $tmpdir/*.tmp

cp local/english.glm $dir/glm
#note, although utt2spk contains mappings to the whole meetings for simulatenous scoring
#we need to know which speakers overlap at meeting level, hence we generate an extra utt2spk_stm file
local/convert2stm.pl $dir utt2spk_stm > $dir/stm

utils/validate_data_dir.sh --no-feats $dir

echo AMI $SET set data preparation succeeded.

