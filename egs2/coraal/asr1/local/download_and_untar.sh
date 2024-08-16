#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Xingyu Na
# Apache 2.0

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <download-list-url>"
  echo "e.g.: $0 downloads http://lingtools.uoregon.edu/coraal/coraal_download_list.txt"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  exit 1;
fi

data=$1
download_list_url=$2

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi


if [ -z "$download_list_url" ]; then
  echo "$0: empty download list URL."
  exit 1;
fi

if [ -f $data/.complete ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0;
fi

# TODO: update or remove
# sizes of the archive files in bytes.
sizes="15582913665 1246920"
if [ -f $data/$part.tgz ]; then
  size=$(/bin/ls -l $data/$part.tgz | awk '{print $5}')
  size_ok=false
  for s in $sizes; do if [ $s == $size ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/$part.tgz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm $data/$part.tgz
  else
    echo "$data/$part.tgz exists and appears to be complete."
  fi
fi

if [ ! -f $data/coraal_download_list.txt ]; then
  if ! command -v wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  echo "$0: downloading data list from $download_list_url."

  cd $data || exit 1
  if ! wget --no-check-certificate $download_list_url; then
    echo "$0: error executing wget $download_list_url"
    exit 1;
  fi

  sed -i 's|http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2018.10.06.txt|http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt|g' coraal_download_list.txt
  sed -i 's|http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2018.10.06.txt|http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt|g' coraal_download_list.txt
fi

# TODO: change the condition (wc -l $data == NUMBER)
if [ ! -f $data/$part.tgz ]; then
  if ! command -v wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi

  echo "$0: downloading and untarring data using $download_list_url. Please be patient."

  cd $data || exit 1
  cat coraal_download_list.txt | while read file_url
  do
    if ! wget --no-check-certificate $file_url; then
      echo "$0: error executing wget $file_url"
      exit 1;
    fi

    file=$(basename $file_url)
    if [[ "$file" == *.tar.gz ]] && ! tar -xvf $file; then
      echo "$0: error executing tar -xvf $file"
      exit 1;
    fi

    # remove files starting with "._", such as ._DTA_se1_ag4_f_01_1.wav
    rm ._*

    if $remove_archive; then
      echo "$0: removing $file since --remove-archive option was supplied."
      rm $file
    fi
  done
fi

echo "$0: Successfully downloaded and untarred CORAAL."

touch $data/.complete

exit 0;
