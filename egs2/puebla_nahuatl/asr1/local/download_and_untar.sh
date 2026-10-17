#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
#             2017  Ewald Enzinger
# Apache 2.0

# Adapted from egs/mini_librispeech/s5/local/download_and_untar.sh (commit 1cd6d2ac3a935009fdc4184cb8a72ddad98fe7d9)

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -le 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url> <filename> <part>>"
  echo "e.g.: $0 /export/data/ https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz cv_corpus_v1.tar.gz 0"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
url=$2
filename=$3
part=$4
filepath="$data/$filename"
workspace=$PWD

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL."
  exit 1;
fi

if [ -f $data/$filename.complete ]; then
  echo "$0: data was already successfully extracted, nothing to do."
  exit 0;
fi


if [ -f $filepath ]; then
  size=$(/bin/ls -l $filepath | awk '{print $5}')
  size_ok=false
  if [ "$filesize" -eq "$size" ]; then size_ok=true; fi;
  if ! $size_ok; then
    echo "$0: removing existing file $filepath because its size in bytes ($size)"
    echo "does not equal the size of the archives ($filesize)."
    rm $filepath
  else
    echo "$filepath exists and appears to be complete."
  fi
fi

if [ ! -f $filepath ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  echo "$0: downloading data from $url.  This may take some time, please be patient."

  cd $data

  file_list=""

  if [ x$part != x ]; then
      for x in $(seq 0 $part); do
          echo "downloading ${url}${x}"
          if ! wget -N --no-check-certificate ${url}${x}; then
            echo "$0: error executing wget ${url}${x}"
            exit 1;
          fi
          file_list=${file_list}" ${filename}${x}"
      done
      echo "start combining"
      cat ${file_list} > ${filename}
  else
      if ! wget -N --no-check-certificate $url; then
        echo "$0: error executing wget $url"
        exit 1;
      fi
  fi

  cd $workspace
fi

cd $data

if ! tar -xzf $filename; then
  echo "$0: error un-tarring archive $filepath"
  exit 1;
fi

cd $workspace

touch $data/$filename.complete

echo "$0: Successfully downloaded and un-tarred $filepath"

if $remove_archive; then
  echo "$0: removing $filepath file since --remove-archive option was supplied."
  rm $filepath
fi
