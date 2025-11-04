#!/usr/bin/env bash

db_root=$1
HF_TOKEN="" # Put your HuggingFace token here

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root>"
    exit 1
fi

if [ -z "${HF_TOKEN}" ]; then
    echo "Please put your HuggingFace token in local/data_download.sh"
    exit 1
fi

cwd=$(pwd)
echo "Downloading Emilia 100 corpus..."
if [ ! -e "${db_root}/emilia_100/.complete" ]; then
    mkdir -p "${db_root}/emilia_100/"
    cd "${db_root}/emilia_100/" || exit 1;
    # Only download 2 files (about ~140 hours) for quick experiments
    for file_index in {0..1}; do
        data_url="https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main/Emilia/EN/EN-B00000${file_index}.tar"
        wget --header="Authorization: Bearer ${HF_TOKEN}" "${data_url}"
        mkdir -p "EN-B00000${file_index}"
        tar -xvf "EN-B00000${file_index}.tar" -C "EN-B00000${file_index}"
        rm "EN-B00000${file_index}.tar"
    done
    cd "${cwd}" || exit 1;
    touch "${db_root}/emilia_100/.complete"
else
    echo "Already exists. Skipped."
fi

echo "Downloading VCTK corpus for test set..."
if [ ! -e "${db_root}/VCTK-Corpus/" ]; then
    mkdir -p "${db_root}"
    cd "${db_root}" || exit 1;
    wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz
    tar xvzf ./VCTK-Corpus.tar.gz
    rm ./VCTK-Corpus.tar.gz
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

echo "Downloading VCTK auxiliary labels..."
if [ ! -e "${db_root}/VCTK-Corpus/lab" ]; then
    cd "${db_root}" || exit 1;
    git clone https://github.com/kan-bayashi/VCTKCorpusFullContextLabel.git
    cp -r VCTKCorpusFullContextLabel/lab ./VCTK-Corpus
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded label data."
else
    echo "Already exists. Skipped."
fi
        