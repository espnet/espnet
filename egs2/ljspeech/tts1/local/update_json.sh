#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

verbose=0

. utils/parse_options.sh
set -e
set -u

if [ $# != 2 ]; then
    echo "Usage: $0 <json> <scp>";
    exit 1;
fi

json=$1
feat=$2
dir=$(dirname ${json})
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -f ${tmpdir}/*.scp

# make ilen.scp
if [ ! -e "$(dirname ${feat})"/utt2num_frames ]; then
    feat-to-len scp:${feat} ark,t:${tmpdir}/ilen.scp
else
    cp "$(dirname ${feat})"/utt2num_frames ${tmpdir}/ilen.scp
fi

# feats scp
cat ${feat} > ${tmpdir}/feat.scp

# idim scp
feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp

# convert to json
rm -f ${tmpdir}/*.json
for x in ${tmpdir}/feat.scp ${tmpdir}/idim.scp ${tmpdir}/ilen.scp; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir}/${k}.json
done

# add to json
addjson.py --verbose ${verbose} \
    ${json} ${tmpdir}/feat.json ${tmpdir}/idim.json ${tmpdir}/ilen.json > ${tmpdir}/data.json
mkdir -p ${dir}/.backup
echo "json updated. original json is kept in ${dir}/.backup."
cp ${json} ${dir}/.backup/$(basename ${json})
cp ${tmpdir}/data.json ${json}
rm -rf ${tmpdir}
