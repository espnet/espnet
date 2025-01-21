#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <kaldi_folder> <dataset_name>"
    echo " e.g.: $0 ../../kising/svs1/dump/raw kising"
    exit 1
fi

for x in $(ls ${kaldi_folder}); do
    if [ ! -d ${kaldi_folder}/${x} ]; then
        continue
    fi

    if [ -f ${kaldi_folder}/${x}/utt2spk ]; then
        echo "Fixing spk2utt and utt2spk in ${kaldi_folder}/${x}"
        ./local/fix_dataset_spk.sh ${kaldi_folder}/${x} ${dataset_name}
    elif [ $x == "org" ]; then
        for org_folder in $(ls ${kaldi_folder}/${x}); do
            if [ -f ${kaldi_folder}/${x}/${org_folder}/utt2spk ]; then
                echo "Fixing spk2utt and utt2spk in ${kaldi_folder}/${x}/${org_folder}"
                ./local/fix_dataset_spk.sh ${kaldi_folder}/${x}/${org_folder} ${dataset_name}
            fi
        done
    fi
done
