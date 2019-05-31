#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
unpaired=""
verbose=0

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
rm -f ${tmpdir}/*.scp

# input, which is not necessary for decoding mode, and make it as an option
if [[ "$unpaired" == "feat" || $unpaired == "" || $unpaired == "sae" ]]; then
    if [ -z ${feat} ]; then
        if [ ${verbose} -eq 0 ]; then
            utils/data/get_utt2num_frames.sh ${dir} &> /dev/null
            cp ${dir}/utt2num_frames ${tmpdir}/ilen.scp
            feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp &> /dev/null
        else
            utils/data/get_utt2num_frames.sh ${dir}
            cp ${dir}/utt2num_frames ${tmpdir}/ilen.scp
            feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp
        fi
    fi
    # feats
    cat ${feat} > ${tmpdir}/feat.scp
fi

# output
if [[ "$unpaired" == "text" || $unpaired == "" ]]; then
    if [ -z ${bpecode} ]; then
        paste -d " " <(awk '{print $1}' ${dir}/text) <(cut -f 2- -d" " ${dir}/text | spm_encode --model=${bpecode} --output_format=piece) > ${tmpdir}/token.scp
    elif [ -z ${nlsyms} ]; then
        text2token.py -s 1 -n 1 -l ${nlsyms} ${dir}/text > ${tmpdir}/token.scp
    else
        text2token.py -s 1 -n 1 ${dir}/text > ${tmpdir}/token.scp
    fi
    utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} < ${tmpdir}/token.scp > ${tmpdir}/tokenid.scp
    awk '{print $1 " " NF-1}' < ${tmpdir}/tokenid.scp > ${tmpdir}/olen.scp 
    # +2 comes from CTC blank and EOS
    vocsize=$(tail -n 1 ${dic} | awk '{print $2}')
    odim=$(echo "$vocsize + 2" | bc)
    awk -v odim=${odim} '{print $1 " " odim}' ${dir}/text > ${tmpdir}/odim.scp

    # others
    if [ -z ${lang} ]; then
        awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text > ${tmpdir}/lang.scp
    fi
fi
# clean previous json if exists
rm -f ${tmpdir}/*.json

if [[ $unpaired == "text" || $unpaired == "" ]]; then
    opts="${dir}/text ${dir}/utt2spk ${tmpdir}/*.scp"
    elif [[ $unpaired == "feat" || $unpaired == "sae" ]]; then
        opts="${dir}/utt2spk ${tmpdir}/*.scp"
fi

for x in $opts; do
    k=$(basename ${x} .scp)
    scp2json.py --key ${k} < ${x} > ${tmpdir}/${k}.json
done
mergejson_asrtts.py --unpaired "$unpaired" --verbose ${verbose} ${tmpdir}/*.json

rm -fr ${tmpdir}
