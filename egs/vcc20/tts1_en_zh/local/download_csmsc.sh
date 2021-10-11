#!/usr/bin/env bash
set -e

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

# download dataset
cwd=$(pwd)
if [ ! -e ${db}/CSMSC ]; then
    mkdir -p ${db}
    cd ${db}
    wget https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar
    mkdir CSMSC && cd CSMSC && unrar x ../BZNSYP.rar
    # convert new line code
    find ./PhoneLabeling -name "*.interval" | while read -r line; do
        nkf -Lu -w --overwrite ${line}
    done
    rm ../BZNSYP.rar
    cd ${cwd}
    echo "Successfully finished download."
else
    echo "Already exists. Skip download."
fi
