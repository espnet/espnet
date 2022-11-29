#!/usr/bin/env bash
# transform MAGICDATA to kaldi format

. ./path.sh || exit 1;

tmp=
dir=

if [ $# != 3 ]; then
  echo "Usage: $0 <corpus-data-dir> <tmp-dir> <output-dir>"
  echo " $0 /export/magicdata/wav/train data/local/train data/train"
  exit 1;
fi

corpus=$1
tmp=$2
dir=$3

echo "prepare_data.sh: Preparing data in $corpus"

mkdir -p $tmp
mkdir -p $dir

# corpus check
if [ ! -d $corpus ] || [ ! -f $corpus/wav.scp ] || [ ! -f $corpus/TRANS.txt ]; then
  echo "Error: $0 requires wav.scp and TRANS.txt under $corpus directory."
  exit 1;
fi

awk '{print $1}' $corpus/wav.scp > $tmp/wav_utt.list
# remove the table header in TRANS.txt ("UtteranceID SpeakerID Transcription")
awk '{print $1}' $corpus/TRANS.txt | grep -v 'UtteranceID' > $tmp/trans_utt.list
utils/filter_scp.pl -f 1 $tmp/wav_utt.list $tmp/trans_utt.list > $tmp/utt.list

# wav.scp
awk -F'\t' -v path_prefix=$corpus/../../ '{printf("%s\t%s/%s\n",$1,path_prefix,$2)}' $corpus/wav.scp > $tmp/tmp_wav.scp
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 | uniq > $tmp/wav.scp

# text
awk -F'\t' '{printf("%s\t%s\n",$1,$3)}' $corpus/TRANS.txt > $tmp/tmp_text
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_text | sort -k 1 | uniq > $tmp/text.raw
# remove unicode whitespaces
perl -CSDA -plE 's/[^\S\t]/ /g' $tmp/text.raw > $tmp/text

# utt2spk & spk2utt
awk -F'\t' '{printf("%s\t%s\n",$1,$2)}' $corpus/TRANS.txt > $tmp/tmp_utt2spk
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_utt2spk | sort -k 1 | uniq > $tmp/utt2spk
utils/utt2spk_to_spk2utt.pl $tmp/utt2spk | sort -k 1 | uniq > $tmp/spk2utt

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text spk2utt utt2spk; do
  cp $tmp/$f $dir/$f || exit 1;
done

echo "local/prepare_data.sh succeeded"
exit 0;
