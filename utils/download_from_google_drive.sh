#!/bin/bash

# Download zipfile from google drive

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

share_url=$1
download_dir=${2:-"downloads"}
file_ext=${3:-"zip"}

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 <share-url> [<download_dir> <file_ext>]";
   echo "e.g.: $0 https://drive.google.com/open?id=1zF88bRNbJhw9hNBq3NrDg8vnGGibREmg downloads .zip"
   echo "Options:"
   echo "    <download_dir>: directory to save downloaded file. (Default=downloads)"
   echo "    <file_ext>: file extension of the file to be downloaded. (Default=zip)"
   exit 1;
fi

[ ! -e "${download_dir}" ] && mkdir -p "${download_dir}"
tmp=$(mktemp "${download_dir}/XXXXXX.${file_ext}")

# file id in google drive can be obtain from sharing link
# ref: https://qiita.com/namakemono/items/c963e75e0af3f7eed732
file_id=$(echo "${share_url}" | cut -d"=" -f 2)

# define decompressor
decompress () {
    filename=$1
    decompress_dir=$2
    if echo "${filename}" | grep -q ".zip"; then
        unzip "${filename}" -d "${decompress_dir}"
    elif echo "${filename}" | grep -q -e ".tar" -e ".tar.gz" -e ".tgz"; then
        tar xvzf "${filename}" -C "${decompress_dir}"
    else
        echo ${filename}
        echo "Unsupported file extension." >&2 && exit 1
    fi
}

# Try-catch like processing
(
    wget "https://drive.google.com/uc?export=download&id=${file_id}" -O "${tmp}"
    decompress "${tmp}" "${download_dir}"
) || {
    # Do not allow error from here
    set -e
    # sometimes, wget from google drive is failed due to virus check confirmation
    # to avoid it, we need to do some tricky processings
    # see https://stackoverflow.com/questions/20665881/direct-download-from-google-drive-using-google-drive-api
    curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${file_id}" > /tmp/intermezzo.html
    postfix=$(grep -Po 'uc-download-link" [^>]* href="\K[^"]*' /tmp/intermezzo.html | sed 's/\&amp;/\&/g')
    curl -L -b /tmp/cookies "https://drive.google.com${postfix}" > "${tmp}"
    decompress "${tmp}" "${download_dir}"
}

# remove tmpfiles
rm "${tmp}"
echo "Sucessfully downloaded zip file from ${share_url}"
