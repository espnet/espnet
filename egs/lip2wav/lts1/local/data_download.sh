#!/bin/bash -e

# Copyright 2019 Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

for spk in chess dl eh hs; do
    spk_dir=${db}/lip2wav_dataset/${spk}
    #if [ ! -e ${spk_dir} ]; then
        for dataset in train dev test; do
            mkdir -p ${spk_dir}/${dataset}
            echo "Downloading ${dataset}-set of ${spk}"
            youtube-dl -f best -a filelists/${spk}/${dataset}.txt -o ${spk_dir}/${dataset}/"%(id)s.%(ext)s"
        done

        for dataset in train dev test; do
            mkdir -p ${spk_dir}/${dataset}/audio 
            mkdir -p ${spk_dir}/${dataset}/video
            for file in $(find -L ${spk_dir}/${dataset} -mindepth 1 -maxdepth 1 -type f -iname "*.mp4" | sort); do
                echo $file
                fname=$(basename $file .mp4)
                ffmpeg -loglevel panic -i $file -f segment -an -vcodec copy \
                    -segment_time 30 "$spk_dir/${dataset}/video/${fname}-%d.mp4"
                ffmpeg -loglevel panic -i $file -f segment -vn -acodec pcm_s16le -ac 1 -ar 16000 \
                    -segment_time 30 "$spk_dir/${dataset}/audio/${fname}-%d.wav"
                rm $file
            done
        done

    #else
    #    echo "${spk}'s data already exists, skip download."
    #fi
done
