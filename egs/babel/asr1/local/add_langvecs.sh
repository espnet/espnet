#!/bin/bash

cmd=run.pl
nj=20
. ./conf/lang.conf

feat_type="phon+inv"

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/add_langvecs.sh <odir> <feats (scp)> <corpus_id>"
  exit 1;
fi  

odir=$1
feats=$2
corpus_id=$3

featname=${feats%%.scp}
featname=`basename ${featname}`
mkdir -p ${odir}/{split${nj}/log,log}

$cmd JOB=1:$nj ${odir}/split${nj}/log/split.JOB.log \
  utils/split_scp.pl -j $nj \$\[JOB -1\] ${feats} ${odir}/split${nj}/${featname}.JOB.scp 

$cmd JOB=1:$nj ${odir}/log/feat.JOB.log \
  python ./local_/add_langvecs.py --feature ${feat_type} \
    ${odir}/feats.JOB ${odir}/split${nj}/${featname}.JOB.scp ${lang_vecs}

mv ${feats} ${feats}.bk
cat ${odir}/feats.*.scp > ${feats}
