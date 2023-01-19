#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
dprep_stage=0
stage=1

# paths
chime5_root=
chime6_root=
dipco_root=
mixer6_root=
manifests_dir=${PWD}/dataset
gss_dump_dir=
tr_dsets=tr
valid_dsets=cv
infer_dsets=

# gss options


# asr options

. ./path.sh
. ./cmd.sh


if [ ${stage} -le 0 ]; then
  # create the dataset


fi


if [ ${stage} -le 2 ]; then
  # check if gss is installed, if not stop, user must manually install it
  if ! command -v gss &> /dev/null
    then
      echo "GPU-based Guided Source Separation (GSS) could not be found,
      please refer to the README for how to install it. \n
      See also https://github.com/desh2608/gss for more informations."
      exit
  fi

  for dset in tr cv tt: do
    $cmd JOB=1:$nj $EXP_DIR/$part/log/enhance.JOB.log \
    gss enhance cuts \
        $EXP_DIR/$dset/cuts.jsonl.gz $EXP_DIR/$dset/split$nj/cuts_per_segment.JOB.jsonl.gz \
        $EXP_DIR/$dset/enhanced \
        --bss-iterations 20 \
        --context-duration 15.0 \
        --use-garbage-class \
        --min-segment-length 0.0 \
        --max-segment-length 20.0 \
        --max-batch-duration 30.0 \
        --max-batch-cuts 1 \
        --num-buckets 4 \
        --num-workers 4 \
        --force-overwrite \
        --duration-tolerance 3.0 || exit 1
  done
fi





if [ ${stage} -le 2 ]; then
  # asr training
  # dump lhotse manifests to kaldi manifests and concatenate with GSS ones


fi





