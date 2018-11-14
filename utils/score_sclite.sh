#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""

. utils/parse_options.sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data-dir> <dict> <output_type>";
    exit 1;
fi

dir=$1
dic=$2
output_type=$3

reftrn=ref.${output_type}.trn
hyptrn=hyp.${output_type}.trn

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn.py ${dir}/data.json ${dic} ${dir}/${reftrn} ${dir}/${hyptrn} ${output_type}

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/${hyptrn}
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/${reftrn} ${dir}/${reftrn}.org
    cp ${dir}/${hyptrn} ${dir}/${hyptrn}.org
    filt.py -v $nlsyms ${dir}/${reftrn}.org > ${dir}/${reftrn}
    filt.py -v $nlsyms ${dir}/${hyptrn}.org > ${dir}/${hyptrn}
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/${hyptrn}
    sed -i.bak3 -f ${filter} ${dir}/${reftrn}
fi

sclite -r ${dir}/${reftrn} trn -h ${dir}/${hyptrn} trn -i rm -o all stdout > ${dir}/result.${output_type}.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    # If we're using phn output type and also requesting WER, throw an error.
    if [ $output_type == "phn"]; then
        echo "Can't do WER evaluation with the phoneme output type."
        exit 1
    fi

    if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/${reftrn} | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/${hyptrn} | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    else
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/${reftrn} > ${dir}/ref.wrd.trn
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/${hyptrn} > ${dir}/hyp.wrd.trn
    fi
    sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt
	
    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
