#!/bin/bash

# Copyright 2019 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ $# != 1 ]; then
  echo "Usage: $0 <dataset-dir>"
  exit 1
fi

dir=$1

mkdir -p ${dir}

if [[ ! -f ${dir}/public_meta_data_v03.csv || $(stat -c %s ${dir}/public_meta_data_v03.csv) != "1468397698" ]]; then
  rm -rf ${dir}/public_meta_data_v03.csv
  wget -O ${dir}/public_meta_data_v03.csv https://ru-open-stt.ams3.digitaloceanspaces.com/public_meta_data_v03.csv
fi

if [[ ! -f ${dir}/bad_trainval_v03.csv || $(stat -c %s ${dir}/bad_trainval_v03.csv) != "660869" ]]; then
  rm -rf ${dir}/bad_trainval_v03.csv
  wget -O - https://github.com/snakers4/open_stt/files/3177895/bad_trainval_v03.zip | bsdtar -x -f- -C ${dir}
fi

if [[ ! -f ${dir}/bad_public_train_v03.csv || $(stat -c %s ${dir}/bad_public_train_v03.csv) != "36939866" ]]; then
  rm -rf ${dir}/bad_public_train_v03.csv
  wget -O - https://github.com/snakers4/open_stt/files/3177907/bad_public_train_v03.zip | bsdtar -x -f- -C ${dir}
fi

if [[ ! -f ${dir}/share_results_v02.csv || $(stat -c %s ${dir}/share_results_v02.csv) != "22046995" ]]; then
  rm -rf ${dir}/share_results_v02.csv
  wget -O - https://github.com/snakers4/open_stt/files/3180977/share_results_v02.zip | bsdtar -x -f- -C ${dir}
fi

if [[ ! -d ${dir}/ru_open_stt ]]; then
  echo "Please download the dataset from http://academictorrents.com/details/4a2656878dc819354ba59cd29b1c01182ca0e162 to ${dir} and unpack its *.tar.gz files"
  exit 1
fi
