#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
case=lc.rm

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn.py ${dir}/data.json ${dic} --refs ${dir}/ref.trn --hyps ${dir}/hyp.trn

if ${remove_blank}; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ -n "${nlsyms}" ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v ${nlsyms} ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ -n "${filter}" ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
fi

# case-sensitive WER
if [ ${case} = tc ]; then

  # detokenize
  detokenizer.perl -l en -q < ${dir}/ref.trn > ${dir}/ref.trn.detok
  detokenizer.perl -l en -q < ${dir}/hyp.trn > ${dir}/hyp.trn.detok

  sclite -s -r ${dir}/ref.trn.detok trn -h ${dir}/hyp.trn.detok trn -i rm -o all stdout > ${dir}/result.tc.txt

  echo "write a case-sensitive CER (or TER) result in ${dir}/result.tc.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.tc.txt

  if ${wer}; then
      if [ -n "$bpe" ]; then
          spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
          spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
      else
          sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
          sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
      fi

      # detokenize
      detokenizer.perl -l en -q < ${dir}/ref.wrd.trn > ${dir}/ref.wrd.trn.detok
      detokenizer.perl -l en -q < ${dir}/hyp.wrd.trn > ${dir}/hyp.wrd.trn.detok

      sclite -s -r ${dir}/ref.wrd.trn.detok trn -h ${dir}/hyp.wrd.trn.detok trn -i rm -o all stdout > ${dir}/result.wrd.tc.txt

      echo "write a case-sensitive WER result in ${dir}/result.wrd.tc.txt"
      grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.tc.txt
  fi
fi

# lowercasing
lowercase.perl < ${dir}/hyp.trn > ${dir}/hyp.trn.lc
lowercase.perl < ${dir}/ref.trn > ${dir}/ref.trn.lc

# remove punctuation
paste -d "(" <(cut -d '(' -f 1 ${dir}/hyp.trn.lc | local/remove_punctuation.pl | sed -e "s/  / /g") <(cut -d '(' -f 2- ${dir}/hyp.trn.lc) > ${dir}/hyp.trn.lc.rm
paste -d "(" <(cut -d '(' -f 1 ${dir}/ref.trn.lc | local/remove_punctuation.pl | sed -e "s/  / /g") <(cut -d '(' -f 2- ${dir}/ref.trn.lc) > ${dir}/ref.trn.lc.rm

# detokenize
detokenizer.perl -l en -q < ${dir}/ref.trn.lc.rm > ${dir}/ref.trn.lc.rm.detok
detokenizer.perl -l en -q < ${dir}/hyp.trn.lc.rm > ${dir}/hyp.trn.lc.rm.detok

sclite -r ${dir}/ref.trn.lc.rm.detok trn -h ${dir}/hyp.trn.lc.rm.detok trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    if [ -n "$bpe" ]; then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn.lc.rm | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn.lc.rm
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn.lc.rm | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn.lc.rm
    else
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn.lc.rm > ${dir}/ref.wrd.trn.lc.rm
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn.lc.rm > ${dir}/hyp.wrd.trn.lc.rm
    fi

    # detokenize
    detokenizer.perl -l en -q < ${dir}/ref.wrd.trn.lc.rm > ${dir}/ref.wrd.trn.lc.rm.detok
    detokenizer.perl -l en -q < ${dir}/hyp.wrd.trn.lc.rm > ${dir}/hyp.wrd.trn.lc.rm.detok

    sclite -r ${dir}/ref.wrd.trn.lc.rm.detok trn -h ${dir}/hyp.wrd.trn.lc.rm.detok trn -i rm -o all stdout > ${dir}/result.wrd.txt

    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
