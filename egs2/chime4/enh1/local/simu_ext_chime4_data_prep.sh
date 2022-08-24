#!/usr/bin/env bash
set -e


# Config:
eval_flag=true # make it true when the evaluation data are released
track=6
annotations=
extra_annotations=
. utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement-name> <patched-speech-directory>\n\n" `basename $0`
  echo "The argument should be a the directory that only contains patched speech data."
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
list_set="tr05_simu_$enhan dt05_simu_$enhan et05_simu_$enhan"
else
list_set="tr05_simu_$enhan dt05_simu_$enhan"
fi

cd $dir

if [[ "$track" == "1" ]]; then
  # 1-ch track
  find ${audio_dir}/isolated/ -name '*.wav' | grep 'tr05_bus_simu\|tr05_caf_simu\|tr05_ped_simu\|tr05_str_simu' | sort -u > tr05_simu_$enhan.flist
  if [ ! -f "${annotations}/dt05_simu_1ch_track.list" ]; then
    echo "Error: No such file: ${annotations}/dt05_simu_1ch_track.list"
    exit 1
  fi
  awk -v dir="${audio_dir}/isolated" '{print(dir "/" $1)}' ${annotations}/dt05_simu_1ch_track.list | sort -u > dt05_simu_$enhan.flist
  if $eval_flag; then
    if [ ! -f "${extra_annotations}/et05_simu_1ch_track.list" ]; then
      echo "Error: No such file: ${extra_annotations}/et05_simu_1ch_track.list"
      exit 1
    fi
    awk -v dir="${audio_dir}/isolated" '{print(dir "/" $1)}' ${extra_annotations}/et05_simu_1ch_track.list | sort -u > et05_simu_$enhan.flist
  fi

  # make a scp file from file list
  for x in $list_set; do
    if [[ "$x" =~ tr05* ]]; then
      cat $x.flist | awk -F'[/]' '{print $NF}'| sed -e 's/\.wav/_SIMU/' > ${x}_wav.ids
    else
      cat $x.flist | awk -F'[/]' '{print $NF}'| sed -e 's/\.CH[0-9]\.wav/_SIMU/' > ${x}_wav.ids
    fi
    paste -d" " ${x}_wav.ids $x.flist | sort -k 1 > ${x}_wav.scp
    sed -E "s#${audio_dir}/isolated/(.*).wav#${audio_dir}/isolated_ext/\1.Clean.wav#g" ${x}_wav.scp > ${x}_spk1_wav.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" ${x}_spk1_wav.scp > ${x}_noise_wav.scp
  done

elif [[ "$track" == "6" ]]; then
  # 6-ch track
  for ch in $(seq 1 6); do
    find ${audio_dir}/isolated/ -name "*.CH${ch}.wav" | grep 'tr05_bus_simu\|tr05_caf_simu\|tr05_ped_simu\|tr05_str_simu' | sort -u > tr05_simu_$enhan.CH${ch}.flist
    find ${audio_dir}/isolated/ -name "*.CH${ch}.wav" | grep 'dt05_bus_simu\|dt05_caf_simu\|dt05_ped_simu\|dt05_str_simu' | sort -u > dt05_simu_$enhan.CH${ch}.flist
    if $eval_flag; then
      find ${audio_dir}/isolated/ -name "*.CH${ch}.wav" | grep 'et05_bus_simu\|et05_caf_simu\|et05_ped_simu\|et05_str_simu' | sort -u > et05_simu_$enhan.CH${ch}.flist
    fi

    # make a scp file from file list
    for x in $list_set; do
      cat $x.CH${ch}.flist | awk -F'[/]' '{print $NF}'| sed -e "s/\.CH${ch}\.wav/_SIMU/" > ${x}_wav.CH${ch}.ids
      paste -d" " ${x}_wav.CH${ch}.ids $x.CH${ch}.flist | sort -k 1 > ${x}_wav.CH${ch}.scp
      sed -E "s#${audio_dir}/isolated/(.*)\.CH${ch}.wav#${audio_dir}/isolated_ext/\1.CH${ch}.Clean.wav#g" ${x}_wav.CH${ch}.scp > ${x}_spk1_wav.CH${ch}.scp
    done
  done

  for x in $list_set; do
    # drop the second channel to follow the convention in CHiME-4
    # see P27 in https://hal.inria.fr/hal-01399180/file/vincent_CSL16.pdf
    mix-mono-wav-scp.py ${x}_wav.CH{1,3,4,5,6}.scp > ${x}_wav.scp
    mix-mono-wav-scp.py ${x}_spk1_wav.CH{1,3,4,5,6}.scp > ${x}_spk1_wav.scp
    sed -E "s#\.Clean\.wav#\.Noise\.wav#g" ${x}_spk1_wav.scp > ${x}_noise_wav.scp
  done
fi

# make a transcription from dot
# simulation training data extract dot file from original WSJ0 data
# since it is generated from these data
if [ ! -e dot_files.flist ]; then
  echo "Could not find $dir/dot_files.flist files, first run local/clean_wsj0_data_prep.sh";
  exit 1;
fi
cat tr05_simu_${enhan}_wav.scp | awk -F'[_]' '{print $2}' | tr '[A-Z]' '[a-z]' \
    | $local/find_noisy_transcripts.pl dot_files.flist | cut -f 2- -d" " > tr05_simu_$enhan.txt
cat tr05_simu_${enhan}_wav.scp | cut -f 1 -d" " > tr05_simu_$enhan.ids
paste -d" " tr05_simu_$enhan.ids tr05_simu_$enhan.txt | sort -k 1 > tr05_simu_$enhan.trans1
# dt05 and et05 simulation data are generated from the CHiME4 booth recording
# and we use CHiME4 dot files
cat dt05_simu.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF "_SIMU"}'> dt05_simu_$enhan.ids
cat dt05_simu.dot | sed -e 's/(.*)//' > dt05_simu_$enhan.txt
paste -d" " dt05_simu_$enhan.ids dt05_simu_$enhan.txt | sort -k 1 > dt05_simu_$enhan.trans1
if $eval_flag; then
cat et05_simu.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF "_SIMU"}'> et05_simu_$enhan.ids
cat et05_simu.dot | sed -e 's/(.*)//' > et05_simu_$enhan.txt
paste -d" " et05_simu_$enhan.ids et05_simu_$enhan.txt | sort -k 1 > et05_simu_$enhan.trans1
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
  cp ${x}_noise_wav.scp $odir/$x/noise1.scp || exit 1;
  cp ${x}.txt     $odir/$x/text    || exit 1;
  cp ${x}.spk2utt $odir/$x/spk2utt || exit 1;
  cp ${x}.utt2spk $odir/$x/utt2spk || exit 1;
done

echo "Data preparation succeeded"
