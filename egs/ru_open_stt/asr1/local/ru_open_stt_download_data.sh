#!/usr/bin/env bash

# Copyright 2019-2020 University of Stuttgart (Pavel Denisov)
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
  rm -f ${f}
  wget https://raw.githubusercontent.com/snakers4/open_stt/4bff5470a29dcca5c7175fa3b6fd106c6151b756/${f}
done

perl -p -i -e 's/.*(radio_v4|public_speech|radio_v4_add)_manifest.tar.gz\n//g' md5sum.lst

for f in $(grep -Poh '\S+\.tar\.gz$' md5sum.lst); do
  if [[ ! -f ${f}.done ]]; then
    . download.sh

    for t in $(grep -Poh '\S+\.tar\.gz$' md5sum.lst); do
      tar -x --overwrite -f ${t}
      touch ${t}.done
      rm -f ${t}
    done
  fi
done

if [[ ! -f public_exclude_file_v5.csv || $(stat -c %s public_exclude_file_v5.csv) != "234868365" ]]; then
  rm -f public_exclude_file_v5.csv
  wget -O - https://github.com/snakers4/open_stt/releases/download/v0.5-beta/public_exclude_file_v5.tar.gz | bsdtar -x -f-
fi

if [[ ! -f exclude_df_youtube_1120.csv || $(stat -c %s exclude_df_youtube_1120.csv) != "33797415" ]]; then
  rm -f exclude_df_youtube_1120.csv
  wget -O - https://github.com/snakers4/open_stt/files/3386441/exclude_df_youtube_1120.zip | bsdtar -x -f-
fi

echo "Data is downloaded and extracted to ${dir}"
