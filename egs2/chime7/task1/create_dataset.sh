#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
stage=2
# this script creates and prepares the official Task 1 data from chime6, dipco and mixer6 datasets.

CHIME7Task1_ROOT=./CHiME7Task1
CHIME5_ROOT= # chime5 needs to be downloaded manually
CHiME6_ROOT=./datasets/chime6 # will be created automatically from chime5
# but if you have it already you can use your existing one.
DIPCO_ROOT=./datasets/dipco # this will be automatically downloaded
MIXER6_ROOT=

if [ ${stage} -le 0 ]; then
  # download DiPCO
  if [ -d "$DIPCO_ROOT/DiPCo" ]; then
    echo "$DIPCO_ROOT/DiPCo already exists,
    exiting as I am assuming it has already been downloadeded."
    exit
  fi
  mkdir -p $DIPCO_ROOT
  if ! [ -d ${DIPCO_ROOT}/DiPCo.tgz ]; then
    wget https://s3.amazonaws.com/dipco/DiPCo.tgz -O ${DIPCO_ROOT}/DiPCo.tgz
  fi
  tar -zxf --directory ${DIPCO_ROOT}
fi
exit

if [ ${stage} -le 1 ]; then
  # from CHiME5 create CHiME6
  local/data/generate_chime6_data.sh \
    --cmd "$train_cmd" \
    ${CHIME5_ROOT} \
    ${CHIME6_ROOT}
fi


if [ ${stage} -le 2 ]; then
  if [ -d "$CHIME7Task1_ROOT" ]; then
    echo "${CHIME7Task1_ROOT} already exists, exiting"
    exit
  fi
  python local/prep_data.py $CHIME6_ROOT $DIPCO_ROOT $MIXER6_ROOT $CHIME7Task1_ROOT
fi