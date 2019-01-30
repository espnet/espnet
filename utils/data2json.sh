#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
category=""
out="" # If omitted, write in stdout

. utils/parse_options.sh

if [ $# != 2 ]; then
    cat << EOF 1>&2
Usage: $0 <data-dir> <dict>
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

dir=$1
dic=$2
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

# 1. Create scp files for inputs
#   These are not necessary for decoding mode, and make it as an option
mkdir -p ${tmpdir}/input
if [ -n "${feat}" ]; then
    cat ${feat} > ${tmpdir}/input/feat.scp

    # Dump in the "legacy" style JSON format
    if [ -n "${filetype}" ]; then
        awk -v filetype=${filetype} '{print $1 " " filetype}' ${feat} \
            > ${tmpdir}/input/filetype.scp
    fi

    feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
        --filetype "${filetype}" \
        --preprocess-conf "${preprocess_conf}" \
        --verbose ${verbose} ${feat} ${tmpdir}/input/shape.scp
fi


# 2. Create scp files for outputs
mkdir -p ${tmpdir}/output
if [ -n "${bpecode}" ]; then
    paste -d " " <(awk '{print $1}' ${dir}/text) <(cut -f 2- -d" " ${dir}/text \
        | spm_encode --model=${bpecode} --output_format=piece) \
        > ${tmpdir}/output/token.scp
elif [ -n "${nlsyms}" ]; then
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dir}/text > ${tmpdir}/output/token.scp
else
    text2token.py -s 1 -n 1 ${dir}/text > ${tmpdir}/output/token.scp
fi
< ${tmpdir}/output/token.scp utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/output/tokenid.scp
# +2 comes from CTC blank and EOS
vocsize=$(tail -n 1 ${dic} | awk '{print $2}')
odim=$(echo "$vocsize + 2" | bc)
< ${tmpdir}/output/tokenid.scp awk -v odim=${odim} '{print $1 " " NF-1 "," odim}' > ${tmpdir}/output/shape.scp

cat ${dir}/text > ${tmpdir}/output/text.scp


# 3. Create scp files for the others
mkdir -p ${tmpdir}/other
if [ -n "${lang}" ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text > ${tmpdir}/other/lang.scp
fi

if [ -n "${category}" ]; then
    awk -v category=${category} '{print $1 " " category}' ${dir}/text \
        > ${tmpdir}/other/category.scp
fi
cat ${dir}/utt2spk > ${tmpdir}/other/utt2spk.scp


# 4. Create JSON files from each scp files
rm -f ${tmpdir}/*/*.json
for intype in 'input' 'output' 'other'; do
    for x in "${tmpdir}/${intype}"/*.scp; do
        k=$(basename ${x} .scp)
        < ${x} scp2json.py --key ${k} > ${tmpdir}/${intype}/${k}.json
    done
done

# 5. Merge JSON files into one and output to stdout
if [ -n "${out}" ]; then
    out_opt="-O ${out}"
else
    out_opt=""
fi
mergejson.py --verbose ${verbose} \
    --input-jsons ${tmpdir}/input/*.json \
    --output-jsons ${tmpdir}/output/*.json \
    --jsons ${tmpdir}/other/*.json ${out_opt}

rm -fr ${tmpdir}
