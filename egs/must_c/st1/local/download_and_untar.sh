#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

remove_archive=false

if [ "$1" == --remove-archive ]; then
    remove_archive=true
    shift
fi

if [ $# -ne 3 ]; then
    echo "Usage: $0 [--remove-archive] <data-base> <lang> <version>"
    echo "e.g.: $0 /n/rd11/corpora_8/MUSTC_v1.0/ de"
    echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
lang=$2
version=$3

if [ ! -d "${data}" ]; then
    echo "$0: no such directory ${data}"
    exit 1;
fi

langs="de_es_fr_it_nl_pt_ro_ru_zh"
if [ ! "$(echo ${langs} | grep ${lang})" ]; then
    echo "$0: no such lang ${lang}"
    exit 1;
fi

if [ ${version} = "v1" ]; then
    instructions="Please download the archives from https://mt.fbk.eu/must-c-release-v1-0/ and place them inside ${data}."
elif [ ${version} = "v2" ]; then
    instructions="Please download the archives from https://mt.fbk.eu/must-c-release-v2-0/ and place them inside ${data}. For en-ja and en-zh, you may just download the archive without H5 files."
else
    echo "${version} is not supported now."
    exit 1;
fi

if [ -f ${data}/.complete_en_${lang} ]; then
    echo "$0: data was already successfully extracted, nothing to do."
    exit 0;
fi

if [ ${version} = "v1" ]; then
    tar_path=${data}/MUSTC_v1.0_en-${lang}.tar.gz
elif [ ${version} = "v2" ]; then
    tar_path=${data}/MUSTC_v2.0_en-${lang}.tar.gz
fi


if [ -f ${tar_path} ]; then
    echo "${tar_path} exists and appears to be complete."
fi

if [ ! -f ${tar_path} ]; then
    echo ${instructions}
fi

if ! tar -zxvf ${tar_path} -C ${data}; then
    echo "$0: error un-tarring archive ${tar_path}"
    exit 1;
fi
touch ${data}/.complete_en_${lang}
echo "$0: Successfully downloaded and un-tarred ${tar_path}"

if $remove_archive; then
    echo "$0: removing ${tar_path} file since --remove-archive option was supplied."
    rm ${tar_path}
fi
