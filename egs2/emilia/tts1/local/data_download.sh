#!/usr/bin/env bash

db_root=$1
HF_TOKEN="" # Put your HuggingFace token here
# Define the range of files to download. If you want to download all files, set START_INDEX=0 and END_INDEX=1139.
# For 100h subset, please set END_INDEX=99.
START_INDEX=0
END_INDEX=1139

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
echo "Downloading Emilia corpus..."
if [ ! -e "${db_root}/emilia/.complete" ]; then
    mkdir -p "${db_root}/emilia/"
    cd "${db_root}/emilia/" || exit 1;
    # Download all files 0..1139 with 6-digit zero padding (000000..001139)
    for file_index in $(seq $START_INDEX $END_INDEX); do
        idx=$(printf "%06d" "$file_index")
        filename="EN-B${idx}"
        # Check if the file has already been downloaded
        if [ -e "${filename}/.complete" ]; then
            echo "${filename} already exists. Skipped."
            continue
        fi

        # Download the file from HuggingFace and extract it
        data_url="https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main/Emilia/EN/${filename}.tar"
        echo "Downloading ${data_url} ..."
        wget -c --header="Authorization: Bearer ${HF_TOKEN}" "${data_url}" || { echo "Failed to download ${data_url}"; exit 1; }
        mkdir -p "${filename}"
        tar -xvf "${filename}.tar" -C "${filename}"
        touch "${filename}/.complete"
        rm -f "${filename}.tar"
    done
    cd "${cwd}" || exit 1;
    touch "${db_root}/emilia/.complete"
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
        