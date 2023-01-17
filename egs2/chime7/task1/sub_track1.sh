#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
stage=1

CHIME6_ROOT=
DIPCO_ROOT=
MIXER6_ROOT=
MANIFESTS_DIR=${CWD}/dataset
AUDIO_DUMP_DIR=
dsets=tr cv tt

if [ ${stage} -le 1 ]; then
  # create JSONL annotation and lhotse manifests
  # create also old annotation compatible with CHiME6 challenge for analysis of results
  #TODO if exists already return error
  python local/prep_data.py $CHIME6_ROOT $DIPCO_ROOT $MIXER6_ROOT $MANIFESTS_DIR
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





