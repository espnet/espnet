#!/bin/bash

. ./path.sh
. ./cmd.sh

langs=conf/langs.conf

. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: ./local/prepare_audio_data.sh <datasets>"
  exit 1;
fi

datasets=$1
# Check that lang file exists
[ -f $langs ] || (echo "Expected langs file to exist" && exit 1);

mkdir -p data/

###############################################################################
# For each language create the appropriate data dir (kaldi format)
###############################################################################
langs_list=""
for l in `cat ${langs} | tr "\n" " "`; do
  echo "---------- Language ${l} -------------------"
  langdata=data/${l}
  langs_list="${langdata} ${langs_list}"
  mkdir -p ${langdata}

  # Make wav.scp
  files=( `find -L ${datasets}/${l}/aligned/wav -name *.wav` )
  for f in ${files[@]}; do
    fname=`basename $f`
    fname=${fname%%.wav}   
    echo "${fname} ${f}"
  done | sort > ${langdata}/wav.scp
  
  awk '{print $1" "$1}' ${langdata}/wav.scp > ${langdata}/utt2spk
  ./utils/spk2utt_to_utt2spk.pl ${langdata}/utt2spk > ${langdata}/spk2utt
 
  # Get text and lexicon.
  # We absolutely have to clean these up. The more I look at these the more I
  # I cringe at how awful they are. For now though I am using them as they are.
  cp ${datasets}/${l}/asr_files/transcription_nopunc.txt ${langdata}/text
  
  #./utils/fix_data_dir.sh ${langdata} 
  
  dict_l=data/dict_${l}
  mkdir -p ${dict_l}
  LC_ALL= sed 's/\s*$//g' ${datasets}/${l}/asr_files/pronunciation.lex |\
    sort > ${dict_l}/lexicon.txt
  
  # Make train dev test (80/10/10)
  num_utts=`cat ${langdata}/text | wc -l`
  num_train=$((num_utts * 80 / 100))
  num_heldout=$((num_utts - num_train))
  num_dev=$((num_heldout / 2))

  # Make Dev 
  ./utils/subset_data_dir.sh ${langdata} $num_heldout ${langdata}_ho
  ./utils/subset_data_dir.sh ${langdata}_ho ${num_dev} ${langdata}_dev
  
  # Make Eval
  cp -r ${langdata}_ho ${langdata}_eval
  ./utils/filter_scp.pl --exclude -f 1 ${langdata}_dev/text ${langdata}_ho/text > ${langdata}_eval/text
 
  # Make Train 
  cp -r ${langdata} ${langdata}_train
  ./utils/filter_scp.pl --exclude -f 1 ${langdata}_ho/text ${langdata}/text > ${langdata}_train/text
 
  # Remove corresponding extra utterances in each directory
  for dset in train dev eval; do
    ./utils/fix_data_dir.sh ${langdata}_${dset}
    cat ${langdata}_${dset}/text |\
      ./utils/apply_map.pl -f 2- ${dict_l}/lexicon.txt > ${langdata}_${dset}/text.phn
  done
done


