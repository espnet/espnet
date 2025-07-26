#!/usr/bin/env bash

pip3 install -r local/requirements.txt
python3 local/download.py

IPAPACK_PLUS=$1

if [ -z "${IPAPACK_PLUS}" ]; then
    log "Please pass IPAPACK_PLUS as an argument"
    exit 1
fi

for dataset in $IPAPACK_PLUS/*_shar;
do
    # Extract dataset name by removing "_shar" suffix
    dataset_name=$(basename "$dataset" | sed 's/_shar$//')
    target_dir=$IPAPACK_PLUS/$dataset_name
    echo "processing $dataset_name -> $target_dir"
    mkdir -p $target_dir

    # untar while preserving the structure of the directories
    for audio_tar in $dataset/**/recording*.tar;
    do
        SPLIT=$(cut -d'/' -f3 <<< "$audio_tar")
        mkdir -p $target_dir/$SPLIT
        tar -xf $audio_tar -C $target_dir/$SPLIT/
    done
done
