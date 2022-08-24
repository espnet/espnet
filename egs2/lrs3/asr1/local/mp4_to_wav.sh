#!/usr/bin/env bash


if [ $# -lt 1 ]; then
  echo "Usage: $0 <data-base-path>"
  echo "--args <data-base-path> : The path to the dataset"
  exit 1
fi

data=$1

for dataset in trainval test; do
    for mp4_path in ${data}/${dataset}/*/*.mp4; do
        # Store the .wav file in the same folder where the .mp4 file is
        wav_path=${mp4_path//.mp4/.wav}
        if ! [ -f  ${wav_path} ]; then 
          ffmpeg -y -i ${mp4_path} -loglevel panic -ar 16000 -ac 1 ${wav_path} 
        fi
    done
done
