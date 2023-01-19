#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=-1

# this script creates and prepares the official Task 1 data.

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