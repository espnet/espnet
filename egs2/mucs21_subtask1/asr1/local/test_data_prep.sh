#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /home/IS21_data/ data/test"
  exit 1
fi

src=$1
dst=$2


mkdir -p $dst || exit 1;


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt=$dst/utt;[[ -f "$utt" ]] && rm $utt
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk


for part in Gujarati Hindi Marathi Odia Tamil Telugu;do

	if [ $part == "Gujarati" ];then
		subset="GU"

	elif [ $part == "Hindi" ];then
		subset="HI"

	elif [ $part == "Marathi" ];then
		subset="MR"

	elif [ $part == "Odia" ];then
		subset="OR"

	elif [ $part == "Tamil" ];then
		subset="TA"

	elif [ $part == "Telugu" ];then
		subset="TE"

	fi

	echo "Preparing test data from $src/$part/test"

	sed -e "s/^/${subset}_/" $src/$part/test/transcription.txt >> $trans

	cat $src/$part/test/transcription.txt | awk '{print $1}' | awk '{print s$0, d$0".wav"}' s="$subset"_ d="$src/$part/test/audio/" >> $wav_scp

done


sort $trans | sed 's/[[:space:]]/ /g'>$dst/temp
rm $trans && mv $dst/temp $trans


sort $wav_scp >$dst/temp
rm $wav_scp && mv $dst/temp $wav_scp

#Preparing utt2spk
cat $wav_scp | awk '{print $1}' >$utt
paste $utt $utt > $utt2spk

# Preparing spk2utt
spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

# Validate the test directory
utils/validate_data_dir.sh --no-feats $dst || exit 1;

echo "$0: Successfully prepared data in $dst" ||  exit 1
