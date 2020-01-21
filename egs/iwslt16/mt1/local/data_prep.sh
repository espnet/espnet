#!/bin/bash

# Copyright 2019 Shun Kiyono
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -x
set -e

. ./path.sh || exit 1

data=$1
lang=$2
data_dir=${data}/en-${lang}

# extract training data
cat ${data_dir}/train.tags.en-${lang}.${lang} | grep -v -E "^<" > ${data_dir}/train.raw.${lang}
cat ${data_dir}/train.tags.en-${lang}.en | grep -v -E "^<" > ${data_dir}/train.raw.en


# extract tst2010..4
for NAME in tst2010 tst2011 tst2012 tst2013 tst2014
do
  for LANG in en $lang
  do
  cat ${data_dir}/IWSLT16.TED.${NAME}.en-${lang}.${LANG}.xml | grep -E "^<seg" | grep -E "seg>$" | perl -ne '@col=split(/ /); print("@col[2..$#col-1]\n");' > \
  ${data_dir}/${NAME}.raw.${LANG}
  done
done



