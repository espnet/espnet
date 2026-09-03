#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <audio-data-dir>"
  echo "e.g.: $0 /home/IS21_data/"
  exit 1
fi

src=$1

for part in Gujarati Hindi Marathi Odia Tamil Telugu;do

	[ ! -d "$src/$part" ] && echo "Expected directory $src/$part to exist" && exit 1;

	[ ! -d "$src/$part/train" ] && echo "Expected directory $src/$part/train to exist" && exit 1;
	[ ! -d "$src/$part/train/audio" ] && echo "Expected directory $src/$part/train/audio to exist" && exit 1;
	[ ! -f "$src/$part/train/transcription.txt" ] && echo "Expected file $src/$part/train/transcription.txt to exist" && exit 1;
	num_aud_files_train=`ls $src/$part/train/audio | wc -l`
	num_transcripts_train=`cat $src/$part/train/transcription.txt | wc -l`
	[ $num_aud_files_train -ne $num_transcripts_train ] && echo "Inconsistent #transcripts($num_transcripts_train) and #audio files($num_aud_files_train)" && exit 1

	[ ! -d "$src/$part/test" ] && echo "Expected directory $src/$part/test to exist" && exit 1;
	[ ! -d "$src/$part/test/audio" ] && echo "Expected directory $src/$part/test/audio to exist" && exit 1;
	[ ! -f "$src/$part/test/transcription.txt" ] && echo "Expected file $src/$part/test/transcription.txt to exist" && exit 1;
	num_aud_files_test=`ls $src/$part/test/audio | wc -l`
	num_transcripts_test=`cat $src/$part/test/transcription.txt | wc -l`
	[ $num_aud_files_test -ne $num_transcripts_test ] && echo "Inconsistent #transcripts($num_transcripts_test) and #audio files($num_aud_files_test)" && exit 1

	echo $src/$part - Correct format

done

echo All 6 languages directories in $src are in the correct format || exit 1;
