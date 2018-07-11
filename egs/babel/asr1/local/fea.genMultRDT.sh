#!/bin/bash
# Copyright 2012-2018  Brno University of Technology (Author: Martin Karafiat, Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

mult_rdt=/mnt/matylda3/karafiat/BABEL/tasks/FeatureExtraction.BUTv3_Y4/forward.multRDT.v0.sh
mult_rdt_cfg=/mnt/matylda3/karafiat/BABEL/tasks/FeatureExtraction.BUTv3_Y4/systems/MultRDTv1.ENV
tooldir=~karafiat/AMI/tools

feagenopt=""
scratch=1
manage_task_opts=
skip_check=false

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--scratch scratch_no [def. 1]] <src-data> <data>"
  echo "Instead: $0 $*"
  echo "\$# = $#"
  exit 1
fi

set -euo pipefail

src_data=$1
data=$2

stkdir=$data/stk_feats
mkdir -p $data
$tooldir/dir.LinkToScratch.sh -scratch $scratch $stkdir || true

# CREATE BATCH FILE,
awk -v pwd=$PWD -v cmd="$mult_rdt -system-cfg $mult_rdt_cfg $feagenopt" -v outdir=$stkdir '
{ wavname=$1; Log=outdir "/" wavname ".log";
  if(match(wavname,"/")) {
    print "WARNING: wavname: " wavname " contains /" > "/dev/stderr";
    logdir=gensub("(.*)/.*","\\1","g",Log); system("mkdir -p " logdir)
  }
}
NF==2{wav=$2;        printf "cd %s; if [ ! -e %s.gz ]; then if    %s -bMapToNewName -wavname %s %s %s > %s 2>&1; then gzip %s; fi; fi\n",pwd,Log,cmd,wavname,wav,outdir,Log,Log}
NF>2 {$1=""; wav=$0; printf "cd %s; if [ ! -e %s.gz ]; then if %s %s -bMapToNewName -wavname %s -  %s > %s 2>&1; then gzip %s; fi; fi\n",pwd,Log,wav,cmd,wavname,outdir,Log,Log}
' $src_data/wav.scp >$stkdir/makefea.sge

### EXTRACT THE MultRDT FEATURES!
sge_opts="-q all.q@blade0[01256789]* -l mem_free=7G,ram_free=7G,tmp_free=10G" # Because of FFV!
#sge_opts="-q all.q@blade0[01256789]*,all.q@blade1*,all.q@*gpu* -l mem_free=7G,ram_free=7G,tmp_free=10G" # temporarily disabled,

$tooldir/manage_task.sh $manage_task_opts $sge_opts -sync yes $stkdir/makefea.sge
# check that we have all the outputs:
for key in $(awk '{ print $1 }' $src_data/wav.scp); do
  log_gz=$stkdir/${key}.log.gz
  [ ! -f $log_gz ] && echo "Missing $log_gz" && terminate_=true
done
#! $skip_check && ${terminate_:-false} && exit 1

# IMPORT THE STK-FEATURES TO KALDI,
$tooldir/dir.LinkToScratch.sh -scratch $scratch $data/data || true
#local/import_feats_htk.sh --nj 5 --cmd "$train_cmd" $src_data $stkdir $data
local/fea.stk2kaldi.v2.sh --nsplit 5 \
    --data_in $src_data --data $data  --stk_dir $stkdir || exit 1

validate_data_dir.sh $data || fix_data_dir.sh $data
