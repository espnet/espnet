#!/usr/bin/env bash

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -lt 3 ]; then
  echo "Usage: $0 [--remove-archive] <data-base-path> <lrs3-username> <lrs3-password>"
  echo "--args [--remove-archive] (Optional) : Remove tar files after successfully untaring"
  echo "--args <data-base-path> : The path where to download the dataset"
  echo "--args <lrs3-username> : The username required to download the dataset"
  echo "--args <lrs3-password> : The password required to download the dataset"
  echo "If you do not have a username/password, please request from: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html"
  exit 1
fi

data=$1
lrs3_username=$2
lrs3_password=$3
lrs3_base_url=https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/
lrs3_train_val_file=lrs3_trainval.zip
lrs3_test_file=lrs3_test_v0.4.zip

echo "Downloading Train/Val data from ${lrs3_base_url}${lrs3_train_val_file}"

if [ -f ${data}/${lrs3_train_val_file} ]; then
     rm  ${data}/${lrs3_train_val_file}
fi

if ! wget  --user ${lrs3_username} --password ${lrs3_password} -P $data  ${lrs3_base_url}${lrs3_train_val_file} ; then
  echo "$0: error executing wget  --user ${lrs3_username} --password ${lrs3_password} -P $data  ${lrs3_base_url}${lrs3_train_val_file}"
  exit 1
fi

echo "Downloading Test data from ${lrs3_base_url}${lrs3_test_file}"

if [ -f ${data}/${lrs3_test_file} ]; then
     rm  ${data}/${lrs3_test_file}
fi

if ! wget  --user ${lrs3_username} --password ${lrs3_password} -P $data   ${lrs3_base_url}${lrs3_test_file} ; then
  echo "$0: error executing wget  --user ${lrs3_username} --password ${lrs3_password} -P $data   ${lrs3_base_url}${lrs3_test_file}"
  exit 1
fi


if [ -e ${data}/trainval ]; then
    echo "Removing existing files in ${data}/trainval before unzipping"
    rm -rf ${data}/trainval
fi

echo "Un-Zipping Train/Val data from ${data}/${lrs3_train_val_file}"

if ! unzip -qq ${data}/${lrs3_train_val_file} -d ${data}; then
    echo "Failed to unzip ${data}/${lrs3_train_val_file}"
    exit 1
fi


if [ -e ${data}/test ]; then
    echo "Removing existing files in ${data}/test before unzipping"
    rm -rf ${data}/test
fi

echo "Un-Zipping Test data from ${data}/${lrs3_test_file}"

if ! unzip -qq ${data}/${lrs3_test_file} -d ${data}; then
    echo "Failed to unzip ${data}/${lrs3_test_file}"
    exit 1
fi

echo "$0: Successfully downloaded and un-tarred ${data}/${lrs3_train_val_file} and ${data}/${lrs3_test_file}"

if $remove_archive; then
  echo "$0: removing${data}/${lrs3_train_val_file} and  ${data}/${lrs3_test_file} file since --remove-archive option was supplied."
  rm ${data}/${lrs3_train_val_file}
  rm ${data}/${lrs3_test_file}
fi
