#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base>"
  echo "e.g.: $0 /export/corpora4/IWSLT/ http://i13pc106.ira.uka.de/~mmueller/iwslt-corpus.zip"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
url=$2

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1;
fi

if [ -f $data/.complete ]; then
  echo "$0: data was already successfully extracted, nothing to do."
  exit 0;
fi


# sizes of the archive files in bytes.
sizes="55563052"

if [ -f $data/iwslt-corpus.zip ]; then
  size=$(/bin/ls -l $data/iwslt-corpus.zip | awk '{print $5}')
  size_ok=false
  for s in $sizes; do if [ $s == $size ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/iwslt-corpus.zip because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/iwslt-corpus.zip
  else
    echo "$data/iwslt-corpus.zip exists and appears to be complete."
  fi
fi


if [ ! -f $data/iwslt-corpus.zip ]; then
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
fi

cd $data

# if ! tar -xvzf iwslt-corpus.zip; then
if ! unzip iwslt-corpus.zip; then
  echo "$0: error un-tarring archive $data/iwslt-corpus.zip"
  exit 1;
fi

touch $data/.complete

echo "$0: Successfully downloaded and un-tarred $data/iwslt-corpus.zip"

if $remove_archive; then
  echo "$0: removing $data/iwslt-corpus.zip file since --remove-archive option was supplied."
  rm $data/iwslt-corpus.zip
fi
