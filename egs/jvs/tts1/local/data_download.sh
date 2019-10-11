#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. path.sh || exit 1;

db=$1

cwd=`pwd`
if [ ! -e ${db}/jvs_ver1 ]; then
    mkdir -p ${db}
    download_from_google_drive.sh https://drive.google.com/open?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt ${db} zip
    echo "Successfully finished download of corpus."
else
    echo "It seems that corpus is already downloaded. Skipped download."
fi
