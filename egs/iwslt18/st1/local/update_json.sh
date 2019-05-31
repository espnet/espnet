#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=1
cmd=run.pl
nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
verbose=0
filetype=""
preprocess_conf=""
out="" # If omitted, write in stdout

text=""
filter_speed_perturbation=false

. utils/parse_options.sh

if [ $# != 3 ]; then
    cat << EOF 1>&2
Usage: $0 <json> <data-dir> <dict>
e.g. $0 data/train data/lang_1char/train_units.txt
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --feat <feat-scp>                                # feat.scp
  --oov <oov-word>                                 # Default: <unk>
  --out <outputfile>                               # If omitted, write in stdout
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
    exit 1;
fi

set -euo pipefail

json=$1
dir=$2
dic=$3
json_dir=$(dirname ${json})
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
trap 'rm -rf ${tmpdir}' EXIT

if [ -z ${text} ]; then
    text=${dir}/text
fi

# 2. Create scp files for outputs
mkdir -p ${tmpdir}/output
if [ -n "${bpecode}" ]; then
    paste -d " " <(awk '{print $1}' ${text}) <(cut -f 2- -d" " ${text} \
        | spm_encode --model=${bpecode} --output_format=piece) \
        > ${tmpdir}/output/token.scp
elif [ -n "${nlsyms}" ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${text} > ${tmpdir}/output/token.scp
else
    text2token.py -s 1 -n 1 ${text} > ${tmpdir}/output/token.scp
fi
< ${tmpdir}/output/token.scp utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/output/tokenid.scp
cat ${tmpdir}/output/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/output/olen.scp
# +2 comes from CTC blank and EOS
vocsize=$(tail -n 1 ${dic} | awk '{print $2}')
odim=$(echo "$vocsize + 2" | bc)
awk -v odim=${odim} '{print $1 " " odim}' ${text} > ${tmpdir}/output/odim.scp

if ${filter_speed_perturbation}; then
    cat ${text} | grep sp1.0 > ${tmpdir}/output/text.scp
else
    cat ${text} > ${tmpdir}/output/text.scp
fi

# 4. Create JSON files from each scp files
rm -f ${tmpdir}/*/*.json
for intype in 'output'; do
    for x in "${tmpdir}/${intype}"/*.scp; do
        k=$(basename ${x} .scp)
        < ${x} scp2json.py --key ${k} > ${tmpdir}/${intype}/${k}.json
    done
done

# add to json
addjson.py --verbose ${verbose} -i false \
  ${json} ${tmpdir}/output/text.json ${tmpdir}/output/token.json ${tmpdir}/output/tokenid.json ${tmpdir}/output/olen.json ${tmpdir}/output/odim.json > ${tmpdir}/data.json
mkdir -p ${json_dir}/.backup
echo "json updated. original json is kept in ${json_dir}/.backup."
cp ${json} ${json_dir}/.backup/$(basename ${json})
cp ${tmpdir}/data.json ${json}

rm -fr ${tmpdir}
