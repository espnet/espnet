#!/usr/bin/env bash
# Copyright 2021    Indian Institute of Science (author: Sathvik Udupa)
# Apache 2.0

path=data
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
  # echo "Exiting.."
  # exit 1
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
  # if [ ! -e ${DIR}/${lang}.done ]; then
      cd ${DIR}
      mkdir -p ${lang}
      cd ${lang}
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_train[$lang]} train
      cp -r ../microsoftspeechcorpusindianlanguages/${msrdata_test[$lang]} test
      mkdir train/audio
      mkdir test/audio
      $cwd/local/down_sample.sh "$DIR/$lang/train/Audios/*" $DIR/$lang/train/'audio'/
      $cwd/local/down_sample.sh "$DIR/$lang/test/Audios/*" $DIR/$lang/test/'audio'/
      # rm -r train/Audios
      # rm -r test/Audios
      touch ${DIR}/${lang}.done
  # fi
done

# declare -A trainset
# trainset['Hindi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi_train.tar.gz'
# trainset['Marathi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Marathi_train.tar.gz'
# trainset['Odia']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Odia_train.tar.gz'
#
# declare -A testset
# testset['Hindi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi_test.tar.gz'
# testset['Marathi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Marathi_test.tar.gz'
# testset['Odia']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Odia_test.tar.gz'
#
# for lang in Hindi Marathi Odia; do
#   if [ ! -e ${DIR}/${lang}.done ]; then
#       cd ${DIR}
#       mkdir -p ${lang}
#       cd ${lang}
#       wget -O test.zip ${testset[$lang]}
#       tar xf "test.zip"
#       rm test.zip
#       # wget -O train.zip ${trainset[$lang]}
#       # tar xf "train.zip"
#       # rm train.zip
#       cd $cwd
#       echo "Successfully finished downloading $lang data."
#       touch ${DIR}/${lang}.done
#   else
#       echo "$lang data already exists. Skip download."
#   fi
# done
