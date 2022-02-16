#!/usr/bin/env bash
# Copyright 2021    Indian Institute of Science (author: Sathvik Udupa)
# Apache 2.0

path=$1
cwd=`pwd`
DIR=$cwd/$path

if [ -d "$DIR" ]; then
  echo "Found existing folder '${path}'.."
else
  echo $path 'folder not found. Creating directory..'
  mkdir $DIR
fi


if [[ -d "$DIR"/microsoftspeechcorpusindianlanguages/ ]]; then
  echo "'$path/microsoftspeechcorpusindianlanguages' exists on your filesystem."
else
  echo "'microsoftspeechcorpusindianlanguages' folder not found. Download it with the following line of code: "
  echo " "
  echo "azcopy copy '<msr opendata azcopy link>' <local folder path> --recursive"
  echo " "
  echo "Then move the contents of <local folder path> to $DIR without changes."
  echo "The directory structure will then be as follows:"
  echo "-$path"
  echo "   -microsoftspeechcorpusindianlanguages"
  echo "       -ta-in-Train"
  echo "       -te-in-Train"
  echo "       - ..."
  echo "Exiting.."
  exit 1
fi

declare -A msrdata_train
msrdata_train['Tamil']=ta-in-Train
msrdata_train['Telugu']=te-in-Train
msrdata_train['Gujarati']=gu-in-Train

declare -A msrdata_test
msrdata_test['Tamil']=ta-in-Test
msrdata_test['Telugu']=te-in-Test
msrdata_test['Gujarati']=gu-in-Test

for lang in Tamil Telugu Gujarati; do
  if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_train[$lang]} train
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_test[$lang]} test
      mkdir train/audio
      mkdir test/audio
      DIR="$DIR/$lang/train/Audios/*"
      reDir=$DIR/$lang/train/'audio'/
      for i in $DIR; do
          ffmpeg -y  -i "$i" -ar 8000 "$reDir${i##*/}"
      done
      DIR="$DIR/$lang/test/Audios/*"
      reDir=$DIR/$lang/test/'audio'/
      for i in $DIR; do
          ffmpeg -y  -i "$i" -ar 8000 "$reDir${i##*/}"
      done
      rm -r train/Audios
      rm -r test/Audios
      touch ${DIR}/${lang}.done
    else
          echo "$lang data already exists. Skip prep."
    fi
done
