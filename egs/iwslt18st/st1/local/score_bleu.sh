#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
word=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
lc=false


. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <data-dir> <dict> <src-dir> <set>";
    exit 1;
fi

dir=$1
dic=$2
set=$4
src=$3/$set/IWSLT.$set
system=st


concatjson.py ${dir}/data.*.json > ${dir}/data.json
local/json2trn_reorder.py ${dir}/data.json ${dic} ${dir}/hyp.trn $src/FILE_ORDER

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
fi

xml_en=$src/IWSLT.TED.$set.en-de.en.xml
xml_de=$src/IWSLT.TED.$set.en-de.de.xml


### character-level
if [ -z $bpe ]; then
  # clean
  sed -e "s/@@ //g" ${dir}/hyp.trn | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' | perl -nle 'print ucfirst' > ${dir}/hyp.trn.clean

  if ${lc}; then
    # case-insensitive
    multi-bleu.perl -lc ${dir}/ref.trn < ${dir}/hyp.trn > ${dir}/result.txt
  else
    # case-sensitive
    multi-bleu.perl ${dir}/ref.trn < ${dir}/hyp.trn > ${dir}/result.txt
  fi
  echo "write a character-level BLEU result in ${dir}/result.txt"
  cat ${dir}/result.txt
fi


### BPE-level
if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
else
  sed -e "s/ //g" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
	sed -e "s/ //g" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
fi

# clean
sed -e "s/@@ //g" ${dir}/hyp.wrd.trn | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' | perl -nle 'print ucfirst' > ${dir}/hyp.wrd.trn.clean


if ${lc}; then
  # segment hypothesis with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.wrd.trn.clean $system de ${dir}/$set.no-case.sgm normalize 0
  sed -e "/<[^>]*>/d" ${dir}/$set.no-case.sgm > ${dir}/$set.no-case.hyp
  cat ${dir}/$set.no-case.hyp | sed -e "s/&/&amp;/g" | perl local/wrap-xml.perl de $xml_en $system > ${dir}/$set.no-case.xml

  # case-insensitive
  mteval-v14.pl -c -s $xml_en -r $xml_de -t ${dir}/$set.xml > ${dir}/result.wrd.txt
  # multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn > ${dir}/result.wrd.txt
else
  # segment hypothesis with RWTH tool
  segmentBasedOnMWER.sh $xml_en $xml_de ${dir}/hyp.wrd.trn.clean $system de ${dir}/$set.sgm normalize 1
  sed -e "/<[^>]*>/d" ${dir}/$set.sgm > ${dir}/$set.hyp
  cat ${dir}/$set.hyp | sed -e "s/&/&amp;/g" | perl local/wrap-xml.perl de $xml_en $system > ${dir}/$set.xml

  # case-sensitive
  mteval-v14.pl -s $xml_en -r $xml_de -t ${dir}/$set.xml > ${dir}/result.wrd.txt
  # multi-bleu-detok.perl ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn > ${dir}/result.wrd.txt
fi

sed -e "s/^\s*$/_EMPTY_/g" ${dir}/$set.hyp > ${dir}/$set.no-empty.hyp


echo "write a word-level BLUE result in ${dir}/result.wrd.txt"
cat ${dir}/result.wrd.txt
