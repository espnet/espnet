#!/usr/bin/env bash

set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k

use_spk_embedding=false

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
# shellcheck disable=SC2034
srcdir=$2
wsj_full_wav=$3

tr="tr_${min_or_max}_${sample_rate}"
cv="cv_${min_or_max}_${sample_rate}"
tt="tt_${min_or_max}_${sample_rate}"

# remove the trailing slash
wavdir=$(echo "$wavdir" | sed 's:/*$::')

# check if the wav dir exists.
for f in $wavdir/tr $wavdir/cv $wavdir/tt; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

data=./data

for x in tr cv tt; do
  target_folder=$(eval echo \$$x)
  mkdir -p ${data}/$target_folder

  wget --continue -O ${data}/${target_folder}/mix.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/${x}/mix.scp
  wget --continue -O ${data}/${target_folder}/ref.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/${x}/ref.scp
  wget --continue -O ${data}/${target_folder}/aux.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/${x}/aux.scp

  awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' ${data}/${target_folder}/mix.scp | \
    sed -e "s#/export/home/clx214/data/wsj0_2mix/min/#${wavdir}/#g" | sort > ${data}/${target_folder}/wav.scp

  awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' ${data}/${target_folder}/ref.scp | \
    sed -e "s#/export/home/clx214/data/wsj0_2mix/min/#${wavdir}/#g" | sort > ${data}/${target_folder}/spk1.scp

  awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' ${data}/${target_folder}/aux.scp | \
    sed -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_tr_s_8k_all/#${wsj_full_wav}/wsj0/si_tr_s/#g" \
        -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_dt_05_8k/#${wsj_full_wav}/wsj0/si_dt_05/#g" \
        -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_et_05_8k/#${wsj_full_wav}/wsj0/si_et_05/#g" | sort > ${data}/${target_folder}/enroll_spk1.scp

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${target_folder}/wav.scp | sort > ${data}/${target_folder}/utt2spk
  utils/utt2spk_to_spk2utt.pl ${data}/${target_folder}/utt2spk > ${data}/${target_folder}/spk2utt
  rm ${data}/${target_folder}/{mix.scp,ref.scp,aux.scp}
done

# prepare speaker embeddings from enrollment audios
if $use_spk_embedding; then
    wget --continue -O local/voxceleb_resnet34_LM.onnx https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx

    python -m pip install onnxruntime

    for x in "$tr" "$cv" "$tt"; do
        # This takes ~4 hours, ~1 hour, ~30 minutes for tr, cv, tt, respectively.
        python local/prepare_spk_embs_scp.py \
            ${data}/${x}/enroll_spk1.scp \
            --model_path local/voxceleb_resnet34_LM.onnx \
            --outdir ${data}/${x}/spk_embs

        mv ${data}/${x}/enroll_spk1.scp ${data}/${x}/enroll_spk1.bak.scp
        mv ${data}/${x}/spk_embs/embs.scp ${data}/${x}/enroll_spk1.scp
    done
fi

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
# shellcheck disable=SC2045
for x in $(ls ${wsj_full_wav}/links/); do find -L ${wsj_full_wav}/links/$x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for f in si_tr_s si_et_05 si_dt_05; do
  awk '{print $1}' ${f}.scp | ${find_transcripts} dot_files.flist > ${f}.trans1

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  noiseword="<NOISE>"
  <${f}.trans1 ${normalize_transcript} ${noiseword} | sort > ${f}.txt || exit 1;
done

# change to the original path
cd ..

awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {n=split($1, lst, "_"); if(substr(lst[n],0,3) == substr(lst[3],0,3)){utt1=lst[3];}else{utt1=lst[5];} text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${tr}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tr}/text
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {n=split($1, lst, "_"); if(substr(lst[n],0,3) == substr(lst[3],0,3)){utt1=lst[3];}else{utt1=lst[5];} text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/${cv}/wav.scp | awk '{$2=""; print $0}' > ${data}/${cv}/text
awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {n=split($1, lst, "_"); if(substr(lst[n],0,3) == substr(lst[3],0,3)){utt1=lst[3];}else{utt1=lst[5];} text=txt[utt1]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt ${data}/${tt}/wav.scp | awk '{$2=""; print $0}' > ${data}/${tt}/text

rm -r tmp
