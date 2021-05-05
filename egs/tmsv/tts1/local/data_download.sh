#!/bin/bash -e

# Copyright 2019 Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
download_url=...

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

# download dataset
cwd=$(pwd)
if [ ! -e ${db}/TSMV ]; then
    mkdir -p ${db}
    ./download_from_google_drive.sh ${download_url} ${db} zip
    echo "Successfully finished download."
else
    echo "Already exists. Skip download."
fi
