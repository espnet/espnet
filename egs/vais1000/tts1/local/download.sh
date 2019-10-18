#!/bin/bash

# Author: enamoria
# Brief : Download VAIS1000 dataset

db=$1
./path.sh || exit 1
cwd=`pwd`

if [ ! -e ${db}/vais1000 ]; then
    mkdir -p ${db}
    cd ${db}
   	
    download_from_google_drive.sh https://drive.google.com/open?id=1HHhLuYhrkk3J6OJctZvgaSd0UgiaROwG .
    cd $cwd
fi
