#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

verbose=0

. utils/parse_options.sh
set -e
set -u

if [ $# != 3 ]; then
    echo "Usage: $0 <json> <scp>";
    echo "unpaired opts = 'text','audio'"
    exit 1;
fi

json=$1
feat=$2
unpaired=$3
dir=$(dirname ${json})
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -f ${tmpdir}/*.scp

# feats scp
cat ${feat} > ${tmpdir}/feat.scp

# idim scp
touch ${tmpdir}/idim.scp
idim=$(copy-vector --print-args=false scp:${tmpdir}/feat.scp ark,t:- | head -n 1 | wc -w)
idim=$(( idim - 3 ))
cat ${tmpdir}/feat.scp | awk '{print $1 " " '"${idim}"'}' > ${tmpdir}/idim.scp

# convert to json
rm -f ${tmpdir}/*.json
for x in ${tmpdir}/feat.scp ${tmpdir}/idim.scp; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir}/${k}.json
done

# add to json
if [ $unpaired == 'text' ]; then
    addjson.py --is-input False -unpaired ${unpaired} --verbose ${verbose} \
    ${json} ${tmpdir}/feat.json ${tmpdir}/idim.json > ${tmpdir}/data.json
else
    addjson_asrtts.py -unpaired ${unpaired} --verbose ${verbose} \
    ${json} ${tmpdir}/feat.json ${tmpdir}/idim.json > ${tmpdir}/data.json
fi
mkdir -p ${dir}/.backup
echo "json updated. original json is kept in ${dir}/.backup."
cp ${json} ${dir}/.backup/$(basename ${json})
cp ${tmpdir}/data.json ${json}
rm -rf ${tmpdir}
