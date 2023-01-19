#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
stage=2

CHIME5_ROOT=
CHIME6_ROOT=
DIPCO_ROOT=
MIXER6_ROOT=
MANIFESTS_DIR=${CWD}/dataset
AUDIO_DUMP_DIR=
dsets=tr,cv,tt

. ./path.sh
. ./cmd.sh


if [ ${stage} -le 0 ]; then
  # download DiPCO
  if [ -d "$DIPCO_ROOT/DiPCo" ]; then
    echo "$DIPCO_ROOT/DiPCo already exists,
    exiting as I am assuming you already downloaded it."
    exit
  fi

  mkdir -p $DIPCO_ROOT
  if ! [ -d ${DIPCO_ROOT}/DiPCo.tgz ]; then
    wget https://s3.amazonaws.com/dipco/DiPCo.tgz -O ${DIPCO_ROOT}/DiPCo.tgz

  fi
  tar -zxf --directory ${DIPCO_ROOT}
fi

if [ ${stage} -le 1 ]; then
  # from CHiME5 create CHiME6
  local/data/generate_chime6_data.sh \
    --cmd "$train_cmd" \
    ${CHIME5_ROOT} \
    ${CHIME6_ROOT}
fi


if [ ${stage} -le 2 ]; then
  if [ -d "$MANIFESTS_DIR" ]; then
    echo "${MANIFESTS_DIR} already exists, exiting"
    exit
  fi
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





