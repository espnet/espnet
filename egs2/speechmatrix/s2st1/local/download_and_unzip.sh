#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
#             2017  Ewald Enzinger
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url> <filename>"
  echo "e.g.: $0 /export/data/ https://us.openslr.org/resources/108/FR.tgz"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  exit 0;
fi

data=$1
url=$2
filename=$3
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
  if ! wget --no-check-certificate $url; then
    echo "$0: error executing wget $url"
    exit 1;
  fi
  cd $workspace
fi

cd $data

# if ! tar -xzf $filename; then
#   echo "$0: error un-tarring archive $filepath"
#   exit 1;
# fi

# Determine the file extension and perform the corresponding extraction operation
case $filename in
  *.tgz|*.gz)
    if ! gzip -d $filename; then
      echo "$0: error unzipping gzip archive $filepath"
      exit 1;
    fi
    ;;
  *.zip)
    if ! unzip $filename; then
      echo "$0: error unzipping zip archive $filepath"
      exit 1;
    fi
    ;;
  *)
    echo "$0: unsupported archive format $filepath"
    exit 1;
    ;;
esac

cd $workspace

touch $data/$filename.complete

echo "$0: Successfully downloaded and unzipped $filepath"

if $remove_archive; then
  echo "$0: removing $filepath file since --remove-archive option was supplied."
  rm $filepath
fi