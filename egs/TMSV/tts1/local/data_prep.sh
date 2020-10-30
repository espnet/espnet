#!/bin/bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Modifications Copyright 2019  Nagoya University (author: Takenori Yoshimura)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <TMSV-dir> <dst-dir>"
  exit 1
fi

TMSV_dir=$1
dst_dir=$2


mkdir -p $dst_dir || exit 1


wav_scp=$dst_dir/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst_dir/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst_dir/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
train_utt_list=$dst_dir/train_utt_list; [[ -f "$train_utt_list" ]] && rm $train_utt_list
dev_utt_list=$dst_dir/dev_utt_list; [[ -f "$dev_utt_list" ]] && rm $dev_utt_list
eval_utt_list=$dst_dir/eval_utt_list; [[ -f "$eval_utt_list" ]] && rm $eval_utt_list

for spk_dir in $(find -L $TMSV_dir -mindepth 1 -maxdepth 1 -type d | sort); do
  audio_dir=${spk_dir}/audio
  video_dir=${spk_dir}/video
  spk_id=$(basename $spk_dir)
  
  count=0
  num_of_file=$(ls $audio_dir | wc -l)
  find -L $audio_dir/ -iname "*.wav" | sort | while read -r wav_file; do
    id=$(basename $wav_file .wav)
    echo "$id $wav_file" >>$wav_scp
    echo "$id $spk_id" >>$utt2spk
    #make the different utt_id lists for spliting the dataset. 
    if [ $count -lt $(($num_of_file - 10)) ]; then
      echo "$id" >>$train_utt_list
    elif [ $count -lt $(($num_of_file -5)) ]; then
      echo "$id" >>$dev_utt_list
    else
      echo "$id" >>$eval_utt_list
    fi
    let "count++"
  done
  python3 local/make_transcription_per_spk.py -s $spk_id \
    -f $spk_dir/${spk_id}_read.txt -t phoneme >>$trans || exit 1

done


spk2utt=$dst_dir/spk2utt
utils/utt2spk_to_spk2utt.pl $utt2spk >  $spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst_dir || exit 1

echo "$0: successfully prepared data in $dst_dir"

exit 0
