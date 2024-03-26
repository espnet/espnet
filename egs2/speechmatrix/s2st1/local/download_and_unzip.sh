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
  if ! wget --no-check-certificate -L -O "$filename" "$url"; then
    echo "$0: error executing wget $url"
    exit 1;
  fi
  cd $workspace
fi

cd $data

# Determine the file extension and perform the corresponding extraction operation
case $filename in
  *.tgz|*.tar.gz)
    # For .tgz and .tar.gz files, use tar to decompress
    if ! tar -xzf $filename; then
      echo "$0: error un-tarring gzip archive $filepath"
      exit 1;
    fi
    ;;
  *.gz)
    # For .gz files, use gzip to decompress
    if ! gzip -d $filename; then
      echo "$0: error unzipping gzip file $filepath"
      exit 1;
    fi
    ;;
  # *.zip)
  #   # For .zip files, use unzip
  #   if ! unzip $filename; then
  #     echo "$0: error unzipping zip archive $filepath"
  #     exit 1;
  #   fi
  #   ;;
  *)
    # Report unsupported file types but don't exit
    echo "$0: '$filename' does not appear to be a supported archive format. No extraction performed."
    ;;
esac


cd $workspace


if [ -f "$filepath" ]; then
    touch $data/$filename.complete
    echo "$0: Successfully downloaded $filepath"

    if $remove_archive && [[ "$filename" =~ \.(tgz|tar\.gz|gz|tsv\.gz)$ ]]; then
      echo "$0: removing $filepath file since --remove-archive option was supplied."
      rm $filepath
    fi
fi
