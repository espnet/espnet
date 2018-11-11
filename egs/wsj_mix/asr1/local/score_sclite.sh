#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
num_spkrs=2

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
ref_trns=""
hyp_trns=""
for i in `seq 1 1 $num_spkrs`; do
    ref_trns=$ref_trns"${dir}/ref${i}.trn "
    hyp_trns=$hyp_trns"${dir}/hyp${i}.trn "
done
json2trn.py ${dir}/data.json ${dic} --num_spkrs $num_spkrs --refs ${ref_trns} --hyps ${hyp_trns}

for n in `seq 1 1 $num_spkrs`; do
    if $remove_blank; then
        sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp${n}.trn
    fi
    if [ ! -z ${nlsyms} ]; then
        cp ${dir}/ref${n}.trn ${dir}/ref${n}.trn.org
        cp ${dir}/hyp${n}.trn ${dir}/hyp${n}.trn.org
        filt.py -v $nlsyms ${dir}/ref${n}.trn.org > ${dir}/ref${n}.trn
        filt.py -v $nlsyms ${dir}/hyp${n}.trn.org > ${dir}/hyp${n}.trn
    fi
    if [ ! -z ${filter} ]; then
        sed -i.bak3 -f ${filter} ${dir}/hyp${n}.trn
        sed -i.bak3 -f ${filter} ${dir}/ref${n}.trn
    fi
done

for (( i=0; i<$[num_spkrs * num_spkrs]; i++ )); do
    ind_r=`expr $i / $num_spkrs + 1`
    ind_h=`expr $i % $num_spkrs + 1`
    sclite -r ${dir}/ref${ind_r}.trn trn -h ${dir}/hyp${ind_h}.trn trn -i rm -o all stdout > ${dir}/result_r${ind_r}h${ind_h}.txt
done

if [ $num_spkrs -eq 2 ]; then
    echo "write a CER (or TER) result in ${dir}/result.txt"
    compute_perm_free_error.sh --num_spkrs 2 ${dir}/result_r1h1.txt ${dir}/result_r1h2.txt \
        ${dir}/result_r2h1.txt ${dir}/result_r2h2.txt > ${dir}/min_perm_result.json
fi
sed -n '2,4p' ${dir}/min_perm_result.json

if ${wer}; then
    for n in `seq 1 1 $num_spkrs`; do
        if [ ! -z $bpe ]; then
            spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref${n}.trn | sed -e "s/▁/ /g" > ${dir}/ref${n}.wrd.trn
            spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp${n}.trn | sed -e "s/▁/ /g" > ${dir}/hyp${n}.wrd.trn
        else
            sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref${n}.trn > ${dir}/ref${n}.wrd.trn
            sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp${n}.trn > ${dir}/hyp${n}.wrd.trn
        fi
    done
    for (( i=0; i<$[num_spkrs * num_spkrs]; i++ )); do
        ind_r=`expr $i / $num_spkrs + 1`
        ind_h=`expr $i % $num_spkrs + 1`
        sclite -r ${dir}/ref${ind_r}.wrd.trn trn -h ${dir}/hyp${ind_h}.wrd.trn trn -i rm -o all stdout > ${dir}/result_r${ind_r}h${ind_h}.wrd.txt
    done

    if [ $num_spkrs -eq 2 ]; then
        echo "write a WER result in ${dir}/result.wrd.txt"
        compute_perm_free_error.sh --num_spkrs 2 --wrd true ${dir}/result_r1h1.wrd.txt ${dir}/result_r1h2.wrd.txt \
            ${dir}/result_r2h1.wrd.txt ${dir}/result_r2h2.wrd.txt > ${dir}/min_perm_result.wrd.json
    fi
    sed -n '2,4p' ${dir}/min_perm_result.wrd.json
fi
