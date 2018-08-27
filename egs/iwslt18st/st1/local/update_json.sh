#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
oov="<unk>"
bpecode=""
verbose=0

. utils/parse_options.sh
set -e
set -u

if [ $# != 3 ]; then
    echo "Usage: $0 <json> <data-dir> <dict>";
    exit 1;
fi

json=$1
data_dir=$2
dic=$3
json_dir=$(dirname ${json})
# tmpdir=`mktemp -d ${json_dir}/tmp-XXXXX`
mkdir -p ${json_dir}/tmp
tmpdir=${json_dir}/tmp
rm -f ${tmpdir}/*.scp

# output
if [ ! -z ${bpecode} ]; then
    paste -d " " <(awk '{print $1}' ${data_dir}/text) <(cut -f 2- -d" " ${data_dir}/text | spm_encode --model=${bpecode} --output_format=piece) > ${tmpdir}/token.scp
elif [ ! -z ${nlsyms} ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${data_dir}/text > ${tmpdir}/token.scp
else
    text2token.py -s 1 -n 1 ${data_dir}/text > ${tmpdir}/token.scp
fi
cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/tokenid.scp
cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/olen.scp
# +2 comes from CTC blank and EOS
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 2" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${data_dir}/text > ${tmpdir}/odim.scp

# convert to json
rm -f ${tmpdir}/*.json
for x in ${data_dir}/text ${data_dir}/utt2spk ${tmpdir}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | scp2json.py --key ${k} > ${tmpdir}/${k}.json
done

# add to json
addjson.py --verbose ${verbose} -i false \
    ${json} ${tmpdir}/text.json ${tmpdir}/token.json ${tmpdir}/tokenid.json ${tmpdir}/olen.json ${tmpdir}/odim.json > ${tmpdir}/data.json
mkdir -p ${json_dir}/.backup
echo "json updated. original json is kept in ${json_dir}/.backup."
cp ${json} ${json_dir}/.backup/$(basename ${json})
cp ${tmpdir}/data.json ${json}
rm -rf ${tmpdir}
