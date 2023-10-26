#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# Code has been modified by members of SPRING Lab IIT-Madras(formerly Speech Lab) for
# SPRING_INX data.

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 5 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <url-base> <data-name> <language> <version>"
  echo "e.g.: $0 downloads https://website.com/data_path"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: dev, test, train"
  exit 1
fi

data=$1
url=$2
name=$3
lang=$4
version=$5

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1
fi

if [ -z "$url" ]; then
  echo "$0: empty URL base."
  exit 1
fi

if [ -f $data/${name}_${lang}_R${version}/.complete ]; then
  echo " 
'${lang}_R${version}' was already successfully extracted, nothing to do."
  exit 0
fi

if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1
fi
full_url=${url}.tar.gz

if [[ `wget -S --spider $full_url 2>&1 | grep 'HTTP/1.1 200 OK'` ]]; then
	echo "downloading data from :
$full_url.
This may take some time, please be patient."
  
  if [[ -f $data/${name}_${lang}_R${version}.tar.gz ]]; then
    rm $data/${name}_${lang}_R${version}.tar.gz
  fi
    
  if ! wget -P $data/ --no-check-certificate $full_url; then
    echo "$0: error executing wget $full_url"
    exit 1
  fi
else
  echo "
All available versions accounted for.
  "
  exit 1
fi

if ! tar -C $data/ -xvzf $data/${name}_${lang}_R${version}.tar.gz >/dev/null; then
  echo "$0: error un-tarring archive $data/${name}_${lang}_R${version}.tar.gz"
  exit 1
fi

touch $data/${name}_${lang}_R${version}/.complete

echo " 
Successfully downloaded and un-tarred $data/${name}_${lang}_R${version}.tar.gz"

if $remove_archive; then
  echo " 
removing $data/${name}_${lang}_R${version}.tar.gz file since --remove-archive option was supplied.
"
  rm $data/${name}_${lang}_R${version}.tar.gz
fi

echo "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
"

