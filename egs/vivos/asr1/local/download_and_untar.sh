#!/usr/bin/env bash

# Copyright 2019  Hieu-Thi Luong
# Copyright 2016  Allen Guo
# Copyright 2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ $# -ne 2 ]; then
  echo "Usage: $0 <datadir> <url>"
  exit 0;
fi

data=$1
url=$2

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL."
  exit 1;
fi

if [ -d $data/vivos ]; then
  echo "$0: vivos directory already exists in $data"
  exit 0;
fi

if [ -f $data/vivos.tar.gz ]; then
  echo "$data/vivos.tar.gz exists"
fi

if [ ! -f $data/vivos.tar.gz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  echo "$0: downloading data (1.4G) from $url."

  cd $data
  if ! wget --no-check-certificate $url; then
    echo "$0: error executing wget $url"
    exit 1;
  fi
fi

cd $data

if ! tar -xvzf vivos.tar.gz; then
  echo "$0: error un-tarring archive $data/vivos.tar.gz"
  exit 1;
fi

echo "$0: Successfully downloaded and un-tarred $data/vivos.tar.gz"
