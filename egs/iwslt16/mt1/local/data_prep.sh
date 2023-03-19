#!/usr/bin/env bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


. ./path.sh || exit 1

data=$1
src_lang=$2
tgt_lang=$3
data_dir=${data}/${src_lang}-${tgt_lang}

# extract training data
if test "$src_lang" = "en"; then
  cat ${data_dir}/train.tags.${src_lang}-${tgt_lang}.${tgt_lang} | grep -v -E "^<" > ${data_dir}/train.raw.${tgt_lang}
  cat ${data_dir}/train.tags.${src_lang}-${tgt_lang}.${src_lang} | grep -v -E "^<" > ${data_dir}/train.raw.${src_lang}
else
  cat ${data_dir}/train.tags.${tgt_lang}-${src_lang}.${tgt_lang} | grep -v -E "^<" > ${data_dir}/train.raw.${tgt_lang}
  cat ${data_dir}/train.tags.${tgt_lang}-${src_lang}.${src_lang} | grep -v -E "^<" > ${data_dir}/train.raw.${src_lang}
fi



# extract tst2010..4
for NAME in tst2010 tst2011 tst2012 tst2013 tst2014
do
  for LANG in $src_lang $tgt_lang
  do
  if test "$src_lang" = "en"; then
    cat ${data_dir}/IWSLT16.TED.${NAME}.${src_lang}-${tgt_lang}.${LANG}.xml | grep -E "^<seg" | grep -E "seg>$" | perl -ne '@col=split(/ /); print("@col[2..$#col-1]\n");' > ${data_dir}/${NAME}.raw.${LANG}
  else
    cat ${data_dir}/IWSLT16.TED.${NAME}.${tgt_lang}-${src_lang}.${LANG}.xml | grep -E "^<seg" | grep -E "seg>$" | perl -ne '@col=split(/ /); print("@col[2..$#col-1]\n");' > ${data_dir}/${NAME}.raw.${LANG}
  fi
  done
done
