#!/bin/bash

# Copyright 2019 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e

if [ $# != 1 ]; then
  echo "Usage: $0 <dataset-dir>"
  exit 1
fi

dir=$1

mkdir -p ${dir}

cd ${dir}

for f in md5sum.lst download.sh; do
  wget https://raw.githubusercontent.com/snakers4/open_stt/8ad945a30cd4bce4f205c994f2c782a8a9159a53/${f}
done

for f in $(grep -Eo '\S+_mp3\.tar\.gz$' md5sum.lst); do
  if [[ ! -f ${f/_mp3.tar.gz/}/.done ]]; then
    . download.sh

    for t in *_mp3.tar.gz; do
      tar -x --overwrite -f ${t}
      touch ${t/_mp3.tar.gz/}/.done
    done
  fi
done

if [[ ! -f public_exclude_file_v5.csv || $(stat -c %s public_exclude_file_v5.csv) != "234868365" ]]; then
  rm -rf public_exclude_file_v5.csv
  wget -O - https://github.com/snakers4/open_stt/releases/download/v0.5-beta/public_exclude_file_v5.tar.gz | bsdtar -x -f-
fi

if [[ ! -f exclude_df_youtube_1120.csv || $(stat -c %s exclude_df_youtube_1120.csv) != "33797415" ]]; then
  rm -rf exclude_df_youtube_1120.csv
  wget -O - https://github.com/snakers4/open_stt/files/3386441/exclude_df_youtube_1120.zip | bsdtar -x -f-
fi

echo "Data is downloaded and extracted to ${dir}"
echo "You can remove *.tar.gz files from ${dir} in order to save disk space"
