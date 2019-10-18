#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
./path.sh || exit 1
cwd=`pwd`

if [ ! -e ${db}/vais1000 ]; then
    mkdir -p ${db}
    cd ${db}
   	
    download_from_google_drive.sh https://drive.google.com/open?id=1HHhLuYhrkk3J6OJctZvgaSd0UgiaROwG .
    cd $cwd
fi
