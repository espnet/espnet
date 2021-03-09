#!/usr/bin/env bash
set -e


# Config:
eval_flag=true  # make it true when the evaluation data are released
track=6
isolated_6ch_dir=  # used for preparing tr05_real_isolated_1ch_track
. utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement-name> <chime4-speech-directory>\n\n" `basename $0`
  echo "The argument should be a the directory that only contains chime4 speech data."
  exit 1;
fi

if [[ ! " 1 6 " =~ " $track " ]]; then
  echo "Error: \$track must be either 1 or 6: ${track}"
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

enhan=$1
audio_dir=$2

dir=`pwd`/data/local/data
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils
odir=`pwd`/data

if $eval_flag; then
list_set="tr05_real_$enhan dt05_real_$enhan et05_real_$enhan"
else
list_set="tr05_real_$enhan dt05_real_$enhan"
fi

cd $dir

if [[ "$track" == "1" ]]; then
  # 1-ch track
  if [ ! -d "${isolated_6ch_dir}" ]; then
    echo "Error: No such directory: ${isolated_6ch_dir}"
    exit 1;
  fi
  find $isolated_6ch_dir/ -name '*.wav' | grep 'tr05_bus_real\|tr05_caf_real\|tr05_ped_real\|tr05_str_real' | sort -u > tr05_real_$enhan.flist
  find $audio_dir/ -name '*.wav' | grep 'dt05_bus_real\|dt05_caf_real\|dt05_ped_real\|dt05_str_real' | sort -u > dt05_real_$enhan.flist
  if $eval_flag; then
    find $audio_dir/ -name '*.wav' | grep 'et05_bus_real\|et05_caf_real\|et05_ped_real\|et05_str_real' | sort -u > et05_real_$enhan.flist
  fi

  # make a scp file from file list
  for x in $list_set; do
    cat $x.flist | awk -F'[/]' '{print $NF}'| sed -e 's/\.wav/_REAL/' > ${x}_wav.ids
    paste -d" " ${x}_wav.ids $x.flist | sort -k 1 > ${x}_wav.scp
    sed -E "s#isolated_1ch_track/(.*)\.wav#isolated_6ch_track/\1.CH0.wav#g" ${x}_wav.scp > ${x}_spk1_wav.scp
  done

elif [[ "$track" == "6" ]]; then
  # 6-ch track
  for ch in $(seq 1 6); do
    find ${audio_dir}/ -name "*.CH${ch}.wav" | grep 'tr05_bus_real\|tr05_caf_real\|tr05_ped_real\|tr05_str_real' | sort -u > tr05_real_$enhan.CH${ch}.flist
    find ${audio_dir}/ -name "*.CH${ch}.wav" | grep 'dt05_bus_real\|dt05_caf_real\|dt05_ped_real\|dt05_str_real' | sort -u > dt05_real_$enhan.CH${ch}.flist
    if $eval_flag; then
      find ${audio_dir}/ -name "*.CH${ch}.wav" | grep 'et05_bus_real\|et05_caf_real\|et05_ped_real\|et05_str_real' | sort -u > et05_real_$enhan.CH${ch}.flist
    fi

    # make a scp file from file list
    for x in $list_set; do
      cat $x.CH${ch}.flist | awk -F'[/]' '{print $NF}'| sed -e "s/\.CH${ch}\.wav/_REAL/" > ${x}_wav.CH${ch}.ids
      paste -d" " ${x}_wav.CH${ch}.ids $x.CH${ch}.flist | sort -k 1 > ${x}_wav.CH${ch}.scp
    done
  done

  for x in $list_set; do
    sed -E "s#${audio_dir}/(.*)\.CH1.wav#${audio_dir}/\1.CH0.wav#g" ${x}_wav.CH1.scp > ${x}_spk1_wav.scp
    mix-mono-wav-scp.py ${x}_wav.CH{1,3,4,5,6}.scp > ${x}_wav.scp
  done
fi

#make a transcription from dot
cat tr05_real.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF "_REAL"}'> tr05_real_$enhan.ids
cat tr05_real.dot | sed -e 's/(.*)//' > tr05_real_$enhan.txt
paste -d" " tr05_real_$enhan.ids tr05_real_$enhan.txt | sort -k 1 > tr05_real_$enhan.trans1
if [[ "$track" != "6" ]]; then
  mv tr05_real_$enhan.trans1 tr05_real_$enhan.trans1.6ch
  awk 'FNR==NR{k=$1; $1=""; a[k]=$0; next}{split($0,lst,"."); print $1, a[lst[1] "_REAL"]}' tr05_real_$enhan.trans1.6ch tr05_real_${enhan}_wav.scp > tr05_real_$enhan.trans1
fi
cat dt05_real.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF "_REAL"}'> dt05_real_$enhan.ids
cat dt05_real.dot | sed -e 's/(.*)//' > dt05_real_$enhan.txt
paste -d" " dt05_real_$enhan.ids dt05_real_$enhan.txt | sort -k 1 > dt05_real_$enhan.trans1
if $eval_flag; then
  cat et05_real.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF "_REAL"}'> et05_real_$enhan.ids
  cat et05_real.dot | sed -e 's/(.*)//' > et05_real_$enhan.txt
  paste -d" " et05_real_$enhan.ids et05_real_$enhan.txt | sort -k 1 > et05_real_$enhan.trans1
fi

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in $list_set;do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done

# Make the utt2spk and spk2utt files.
for x in $list_set; do
  cat ${x}_wav.scp | awk -F'_' '{print $1}' > $x.spk
  cat ${x}_wav.scp | awk '{print $1}' > $x.utt
  paste -d" " $x.utt $x.spk > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

# copying data to data/...
for x in $list_set; do
  mkdir -p $odir/$x
  cp ${x}_wav.scp $odir/$x/wav.scp || exit 1;
  cp ${x}_spk1_wav.scp $odir/$x/spk1.scp || exit 1;
  cp ${x}.txt     $odir/$x/text    || exit 1;
  cp ${x}.spk2utt $odir/$x/spk2utt || exit 1;
  cp ${x}.utt2spk $odir/$x/utt2spk || exit 1;
done

echo "Data preparation succeeded"
