#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <part>"
  echo "e.g.: $0 /export/corpora4/IWSLT/ http://i13pc106.ira.uka.de/~mmueller/iwslt-corpus.zip"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
part=$2

train_url=http://i13pc106.ira.uka.de/~mmueller/iwslt-corpus.zip
if [ $part = "dev2010" ];then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.dev2010.en-de.tgz
elif [ $part = "tst2010" ];then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2010.en-de.tgz
elif [ $part = "tst2013" ];then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2013.en-de.tgz
elif [ $part = "tst2014" ];then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2014.en-de.tgz
elif [ $part = "tst2015" ];then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2015.en-de.tgz
elif [ $part = "tst2018" ];then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2018.en-de.tgz
fi

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

parts="train dev2010 tst2010 tst2013 tst2014 tst2015 tst2018"
if [ ! `echo ${part} | grep ${part}`  ]; then
  echo "$0: no such part $part"
  exit 1
fi


if [ ! $part = tst2018 ]; then
  if [ -f $data/iwslt-corpus/.complete ]; then
    echo "$0: data was already successfully extracted, nothing to do."
    exit 0;
  fi

  # sizes of the archive files in bytes.
  sizes="56896558179"

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
    echo "$0: downloading data from $train_url.  This may take some time, please be patient."

    cd $data
    if ! wget --no-check-certificate -P $data $train_url; then
      echo "$0: error executing wget $train_url"
      exit 1;
    fi
  fi

  cd $data

  if ! unzip $data/iwslt-corpus.zip; then
    echo "$0: error un-tarring archive $data/iwslt-corpus.zip"
    exit 1;
  fi

  touch $data/iwslt-corpus/.complete

  echo "$0: Successfully downloaded and un-tarred $data/iwslt-corpus.zip"

  if $remove_archive; then
    echo "$0: removing $data/iwslt-corpus.zip file since --remove-archive option was supplied."
    rm $data/iwslt-corpus.zip
  fi

else
  if [ -f $data/IWSLT.$part/.complete ]; then
    echo "$0: data was already successfully extracted, nothing to do."
    exit 0;
  fi

  # sizes of the archive files in bytes.
  sizes="262144 524800 524288 524288 524288 1048576"

  if [ -f $data/IWSLT-SLT.$part.en-de.tgz ]; then
    size=$(/bin/ls -l $data/IWSLT-SLT.$part.en-de.tgz | awk '{print $5}')
    # size_ok=false
    # for s in $sizes; do if [ $s == $size ]; then size_ok=true; fi; done
    size_ok=true
    if ! $size_ok; then
      echo "$0: removing existing file $data/IWSLT-SLT.$part.en-de.tgz because its size in bytes $size"
      echo "does not equal the size of one of the archives."
      rm $data/IWSLT-SLT.$part.en-de.tgz
    else
      echo "$data/IWSLT-SLT.$part.en-de.tgz exists and appears to be complete."
    fi
  fi

  if [ ! -f $data/IWSLT-SLT.$part.en-de.tgz ]; then
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

  if ! tar -xvzf IWSLT-SLT.$part.en-de.tgz; then
    echo "$0: error un-tarring archive $data/IWSLT-SLT.$part.en-de.tgz"
    exit 1;
  fi

  touch $data/IWSLT.$part/.complete

  echo "$0: Successfully downloaded and un-tarred $data/IWSLT-SLT.$part.en-de.tgz"

  if $remove_archive; then
    echo "$0: removing $data/IWSLT-SLT.$part.en-de.tgz file since --remove-archive option was supplied."
    rm $data/IWSLT-SLT.$part.en-de.tgz
  fi

fi
