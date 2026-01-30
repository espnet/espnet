#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

. utils/parse_options.sh || exit 1;

if [ $# != 1 ]; then
    echo "usage: $0 datadir"
    exit 1;
fi

dir=$1

cp ${dir}/text ${dir}/text.orig
local/csj_rm_tag.py -s 1 ${dir}/text.orig | sed -e 's/ <sp>//g' > ${dir}/text.tmp
paste -d " " <(cut -f 1 -d" " ${dir}/text.tmp) <(cut -f 2- -d" " ${dir}/text.tmp | tr -d " ") > ${dir}/text
rm ${dir}/text.orig ${dir}/text.tmp
echo "removed tags, space, and <sp>"
