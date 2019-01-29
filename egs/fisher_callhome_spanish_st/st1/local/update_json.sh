#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
oov="<unk>"
bpecode=""
verbose=0

text=""
filter_speed_perturbation=false

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

if [ -z ${text} ]; then
    text=${data_dir}/text
fi

# output
if [ ! -z ${bpecode} ]; then
    paste -d " " <(awk '{print $1}' ${text}) <(cut -f 2- -d" " ${text} | spm_encode --model=${bpecode} --output_format=piece) > ${tmpdir}/token.scp
elif [ ! -z ${nlsyms} ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${text} > ${tmpdir}/token.scp
else
    text2token.py -s 1 -n 1 ${text} > ${tmpdir}/token.scp
fi
cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/tokenid.scp
cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/olen.scp
# +2 comes from CTC blank and EOS
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 2" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${text} > ${tmpdir}/odim.scp

# convert to json
rm -f ${tmpdir}/*.json
if ${filter_speed_perturbation}; then
    cat ${text} | grep sp1.0 | scp2json.py --key text > ${tmpdir}/text.json
else
    cat ${text} | scp2json.py --key text > ${tmpdir}/text.json
fi
for x in ${tmpdir}/*.scp; do
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
