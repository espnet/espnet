#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
#             2017  Ewald Enzinger
# Apache 2.0

# Adapted from egs/mini_librispeech/s5/local/download_and_untar.sh (commit 1cd6d2ac3a935009fdc4184cb8a72ddad98fe7d9)
set -x

remove_archive=false
if [ "$1" == --remove-archive ]; then
    remove_archive=true
    shift
fi

md5=""
if [ "$1" == --md5sum ]; then
    md5=$2
    shift; shift
fi

if [ $# -ne 3 ]; then
    echo "Usage: $0 [--remove-archive] [--md5sum <md5sum>] <data-base> <url> <filename>"
    echo "e.g.: $0 /export/data/ https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz cv_corpus_v1.tar.gz"
    echo "With --remove-archive it will remove the archive after successfully un-tarring it."
    echo "With --md5sum, it will check the sanity of the file using md5sum."
fi

data=$1
url=$2
filename=$3
filepath="$data/$filename"
workspace=$PWD

# Parameter sanity check.
if [ ! -d "$data" ]; then
    echo "$0: no such directory $data"
    exit 1;
fi

if [ -z "$url" ]; then
    echo "$0: empty URL."
    exit 1;
fi

# Cases where it skips downloading
if [ -f $data/$filename.complete ]; then
    echo "$0: data was already successfully extracted, nothing to do."
    exit 0;
fi

need_download=true
if [ -f $filepath ]; then
    if [ -z "${md5}" ]; then
        echo "You need to supply the md5sum via --md5sum option to verify."
        echo "Cowardly stopping the script."
        exit 1
    fi

    echo "Checking md5sum..."
    md5_filepath=$(md5sum $filepath | cut -d " " -f 1)
    if [ $md5 -eq $md5_filepath ]; then
        echo "$filepath exists and its md5sum checks out. Skipping downloading file $filepath."
        need_download=false
    else
        echo "Supplied md5sum does not match. Removing existing file $filepath."
        rm $filepath
    fi
fi

if [ "$need_download" = true ] ; then
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
    cd $workspace
fi

cd $data
if ! tar -xzf $filename; then
    echo "$0: error un-tarring archive $filepath"
    exit 1;
fi

cd $workspace
touch $data/$filename.complete
echo "$0: Successfully downloaded and un-tarred $filepath"

if [ "$remove_archive" = true ] ; then
    echo "$0: removing $filepath file since --remove-archive option was supplied."
    rm $filepath
fi
