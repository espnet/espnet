#!/bin/bash

# Download zipfile from google drive

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

share_url=$1
download_dir=$2

if [ $# -ne 2 ]; then
   echo "Usage: $0 <share-url> <download_dir>";
   echo "e.g.: $0 https://drive.google.com/open?id=1zF88bRNbJhw9hNBq3NrDg8vnGGibREmg downloads"
   exit 1;
fi

[ ! -e "${download_dir}" ] && mkdir -p "${download_dir}"
tmpzip=$(mktemp "${download_dir}/XXXXXX.zip")

# file id in google drive can be obtain from sharing link
# ref: https://qiita.com/namakemono/items/c963e75e0af3f7eed732
file_id=$(echo "${share_url}" | sed -e "s|.*open?id=||g")

# Try-catch like processing
(
    wget "https://drive.google.com/uc?export=download&id=${file_id}" -O "${tmpzip}"
    unzip "${tmpzip}" -d "${download_dir}"
) || {
    # Do not allow error from here
    set -e
    # sometimes, wget from google drive is failed due to virus check confirmation
    # to avoid it, we need to do some tricky processings
    # see https://stackoverflow.com/questions/20665881/direct-download-from-google-drive-using-google-drive-api
    curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${file_id}" > /tmp/intermezzo.html
    postfix=$(grep -Po 'uc-download-link" [^>]* href="\K[^"]*' /tmp/intermezzo.html | sed 's/\&amp;/\&/g')
    curl -L -b /tmp/cookies "https://drive.google.com${postfix}" > "${tmpzip}"
    unzip "${tmpzip}" -d "${download_dir}"
}

# remove tmpfiles
rm "${tmpzip}"
echo "Sucessfully downloaded zip file from ${share_url}"
