#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

remove_archive=false

if [ "$1" == --remove-archive ]; then
    remove_archive=true
    shift
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--remove-archive] <data-base> <set>"
    echo "e.g.: $0 /export/corpora4/IWSLT/ http://i13pc106.ira.uka.de/~mmueller/iwslt-corpus.zip"
    echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
set=$2

train_url=http://i13pc106.ira.uka.de/~mmueller/iwslt-corpus.zip
if [ ${set} = "dev2010" ]; then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.dev2010.en-de.tgz
elif [ ${set} = "tst2010" ]; then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2010.en-de.tgz
elif [ ${set} = "tst2013" ]; then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2013.en-de.tgz
elif [ ${set} = "tst2014" ]; then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2014.en-de.tgz
elif [ ${set} = "tst2015" ]; then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2015.en-de.tgz
elif [ ${set} = "tst2018" ]; then
    url=http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-de/preprocessed/IWSLT-SLT.tst2018.en-de.tgz
fi

if [ ! -d "${data}" ]; then
    echo "$0: no such directory ${data}"
    exit 1;
fi

sets="train dev2010 tst2010 tst2013 tst2014 tst2015 tst2018"
if [ ! `echo ${set} | grep ${set}`  ]; then
    echo "$0: no such set ${set}"
    exit 1;
fi

if [ -f ${data}/${set}/.complete ]; then
    echo "$0: data was already successfully extracted, nothing to do."
    exit 0;
fi

if [ ${set} = train ]; then
    if [ -f ${data}/${set}/iwslt-corpus.zip ]; then
        echo "${data}/${set}/iwslt-corpus.zip exists and appears to be complete."
    fi

    mkdir -p ${data}/${set}

    if [ ! -f ${data}/${set}/iwslt-corpus.zip ]; then
        if ! which wget >/dev/null; then
            echo "$0: wget is not installed."
            exit 1;
        fi
        echo "$0: downloading data from ${train_url}.  This may take some time, please be patient."

        if ! wget --no-check-certificate -P ${data}/${set} ${train_url}; then
            echo "$0: error executing wget ${train_url}"
            exit 1;
        fi
    fi

    if ! unzip ${data}/${set}/iwslt-corpus.zip -d ${data}/${set}; then
        echo "$0: error un-tarring archive ${data}/${set}/iwslt-corpus.zip"
        exit 1;
    fi

    touch ${data}/${set}/.complete
    echo "$0: Successfully downloaded and un-tarred ${data}/${set}/iwslt-corpus.zip"

    if $remove_archive; then
        echo "$0: removing ${data}/${set}/iwslt-corpus.zip file since --remove-archive option was supplied."
        rm ${data}/${set}/iwslt-corpus.zip
    fi
else
    if [ -f ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz ]; then
        echo "${data}/${set}/IWSLT-SLT.${set}.en-de.tgz exists and appears to be complete."
    fi

    mkdir -p ${data}/${set}

    if [ ! -f ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz ]; then
        if ! which wget >/dev/null; then
            echo "$0: wget is not installed."
            exit 1;
        fi
        echo "$0: downloading data from $url.  This may take some time, please be patient."

        if ! wget --no-check-certificate -P ${data}/${set} $url; then
            echo "$0: error executing wget $url"
            exit 1;
        fi
    fi

    if ! tar -xvzf ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz -C ${data}/${set}; then
        echo "$0: error un-tarring archive ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz"
        exit 1;
    fi

    touch ${data}/${set}/.complete
    echo "$0: Successfully downloaded and un-tarred ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz"

    if $remove_archive; then
        echo "$0: removing ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz file since --remove-archive option was supplied."
        rm ${data}/${set}/IWSLT-SLT.${set}.en-de.tgz
    fi
fi
