#!/usr/bin/env bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

remove_archive=false

if [ "$1" == --remove-archive ]; then
    remove_archive=true
    shift
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--remove-archive] <data-base> <lang>"
    echo "e.g.: $0 iwslt16_data de"
    echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
lang=$2


mkdir -p $data

langs="de_fr"
if [ ! $(echo ${langs} | grep ${lang}) ]; then
    echo "$0: no such lang ${lang}"
    exit 1;
fi

if [ ${lang} = "de" ]; then
    url="https://wit3.fbk.eu/archive/2016-01/texts/en/de/en-de.tgz"
fi

if [ ! -f ${data}/en-${lang}.tgz ]; then
    if ! which wget >/dev/null; then
        echo "$0: wget is not installed."
        exit 1;
    fi
    echo "$0: downloading data from ${url}.  This may take some time, please be patient."
    wget $url -O ${data}/en-${lang}.tgz || exit 1
fi

if ! tar -zxvf ${data}/en-${lang}.tgz -C ${data}; then
    echo "$0: error un-tarring archive ${data}/en-${lang}.tgz"
    exit 1;
fi

touch ${data}/.complete_en_${lang}
ln -s ${data}/${lang}-en ${data}/en-${lang}
echo "$0: Successfully downloaded and un-tarred ${data}/en-${lang}.tgz"

if $remove_archive; then
    echo "$0: removing ${data}/en-${lang}.tgz file since --remove-archive option was supplied."
    rm ${data}/en-${lang}.tgz
fi
