#!/bin/bash

# Copyright 2020  Academia Sinica (author: Pin-Jui Ku)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <TMSV-dir> <dst-dir>"
  exit 1
fi

TMSV_dir=$1
dst_dir=$2


mkdir -p $dst_dir || exit 1


wav_scp=$dst_dir/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
video_scp=$dst_dir/video.scp; [[ -f "$video_scp" ]] && rm $video_scp
trans=$dst_dir/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst_dir/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
train_utt_list=$dst_dir/train_utt_list; [[ -f "$train_utt_list" ]] && rm $train_utt_list
dev_utt_list=$dst_dir/dev_utt_list; [[ -f "$dev_utt_list" ]] && rm $dev_utt_list
eval_utt_list=$dst_dir/eval_utt_list; [[ -f "$eval_utt_list" ]] && rm $eval_utt_list

for spk_dir in $(find -L $TMSV_dir -mindepth 1 -maxdepth 1 -type d -iname "SP*" | sort); do
  audio_dir=${spk_dir}/audio
  video_dir=${spk_dir}/video
  spk_id=$(basename $spk_dir)
  
  count=0
  num_of_file=$(ls $audio_dir | wc -l)
  find -L $audio_dir/ -iname "*.wav" | sort | while read -r wav_file; do
    id=$(basename $wav_file .wav)
    echo "$id $wav_file" >>$wav_scp
    echo "$id $spk_id" >>$utt2spk


    # make the different utt_id lists for spliting the dataset.
    # some of the video data are broken, we will remove them when we use the utt_list to split dataset.
    break_flag=0
    for broken_id in SP18_119 SP18_122 SP18_124 SP18_125 SP18_132 SP18_133 SP18_141 SP18_189 SP15_032 SP07_154; do
      if [ $id == $broken_id ]; then
        break_flag=1
      fi
    done
    
    if [ $break_flag -eq 1 ];then
      echo "Remove $id from dataset due to broken video of different fps format. See README.md for details."
      continue        
    elif [ $count -lt $(($num_of_file - 10)) ]; then
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

  find -L $video_dir/ -iname "*.mp4" | sort | while read -r video_file; do
    id=$(basename $video_file .mp4)
    echo "$id $video_file" >>$video_scp
  done

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
