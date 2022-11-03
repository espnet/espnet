#!/usr/bin/env bash

# Copyright 2016  Allen Guo
# Copyright 2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base> <file-name>"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  exit 0;
fi

data=$1
url=$2
filename=$3
folders=(${filename//"."/ })
foldername=${folders[0]}

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -d $data/parla/$foldername ]; then
  echo "$0: parla directory already exists in $data"
  exit 0;
fi

if [ ! -d $data/parla ]; then
  mkdir -p -- "$data/parla"
fi

if [ -f $data/$filename ]; then
  echo "$data/$filename exists"
fi

if [ ! -f $data/$filename ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  full_url=$url
  echo "$0: downloading data (7.7 GB) from $full_url."

  cd $data
  if ! wget --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1;
  fi
fi

cd $data

if ! tar -xvzf $filename -C parla/; then
  echo "$0: error un-tarring archive $data/$filename"
  exit 1;
fi

echo "$0: Successfully downloaded and un-tarred $data/$filename"

if $remove_archive; then
  echo "$0: removing $data/$filename file since --remove-archive option was supplied."
  rm $data/$filename
fi
