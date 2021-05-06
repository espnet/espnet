#!/usr/bin/env bash

set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. ./path.sh

if [ $# -le 2 ]; then
  echo "Arguments should be WSJ0-2MIX directory, the mixing script path and the WSJ0 path, see ../run.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

find_transcripts=$KALDI_ROOT/egs/wsj/s5/local/find_transcripts.pl
normalize_transcript=$KALDI_ROOT/egs/wsj/s5/local/normalize_transcript.pl

wavdir=$1
srcdir=$2
wsj_full_wav=$3

tr="tr_${min_or_max}_${sample_rate}"
cv="cv_${min_or_max}_${sample_rate}"
tt="tt_${min_or_max}_${sample_rate}"

# check if the wav dir exists.
for f in $wavdir/tr $wavdir/cv $wavdir/tt; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

# check if the script file exists.
for f in $srcdir/mix_2_spk_${min_or_max}_tr_mix $srcdir/mix_2_spk_${min_or_max}_cv_mix $srcdir/mix_2_spk_${min_or_max}_tt_mix; do
  if [ ! -f $f ]; then
    echo "Could not find $f.";
    exit 1;
  fi
done

data=./data

for x in tr cv tt; do
  target_folder=$(eval echo \$$x)
  mkdir -p ${data}/$target_folder
  cat $srcdir/mix_2_spk_${min_or_max}_${x}_mix | \
    awk -v dir=$wavdir/$x '{printf("%s %s/mix/%s.wav\n", $1, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${data}/${target_folder}/wav.scp

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${target_folder}/wav.scp | sort > ${data}/${target_folder}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${target_folder}/utt2spk > ${data}/${target_folder}/spk2utt
done

# transcriptions (only for 'max' version)
if [[ "$min_or_max" = "min" ]]; then
  exit 0
fi


# rm -r tmp/ 2>/dev/null
mkdir -p tmp
cd tmp
for i in si_tr_s si_et_05 si_dt_05; do
    cp ${wsj_full_wav}/${i}.scp .
done

# Finding the transcript files:
for x in `ls ${wsj_full_wav}/links/`; do find -L ${wsj_full_wav}/links/$x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for f in si_tr_s si_et_05 si_dt_05; do
  cat ${f}.scp | awk '{print $1}' | ${find_transcripts} dot_files.flist > ${f}.trans1

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  noiseword="<NOISE>"
  cat ${f}.trans1 | ${normalize_transcript} ${noiseword} | sort > ${f}.txt || exit 1;
done

# change to the original path
cd ..

awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text_spk2
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text_spk2
awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text_spk1
awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text_spk2

rm -r tmp
