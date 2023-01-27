#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
stage=2
skip_stages="1"
function contains ()  { [[ $1 =~ (^|[[:space:]])"$2"($|[[:space:]]) ]]; }

# this script creates and prepares the official Task 1 data from chime6, dipco and mixer6 datasets.
CHIME7Task1_ROOT=/tmp/CHiME7Task1
CHIME5_ROOT= # chime5 needs to be downloaded manually
CHIME6_ROOT=/home/samco/dgx/CHiME6/espnet/egs2/chime6/asr1/CHiME6 # will be created automatically from chime5
# but if you have it already you can use your existing one.
DIPCO_ROOT=/home/samco/dgx/CHiME6/DipCO/DiPCo/ # this will be automatically downloaded
MIXER6_ROOT=/home/samco/dgx/mixer6/

if [ ${stage} -le 0 ] && ! contains $skip_stages 0; then
  # download DiPCO

  if [ -d "$DIPCO_ROOT/DiPCo" ]; then
    echo "$DIPCO_ROOT/DiPCo already exists,
    exiting as I am assuming it has already been downloaded."
    exit
  fi
  mkdir -p $DIPCO_ROOT
  if ! [ -d ${DIPCO_ROOT}/DiPCo.tgz ]; then
    wget https://s3.amazonaws.com/dipco/DiPCo.tgz -O ${DIPCO_ROOT}/DiPCo.tgz
  fi
  tar -zxf --directory ${DIPCO_ROOT}
fi

if [ ${stage} -le 1 ] && ! [ contains $skip_stages 1]; then
  # from CHiME5 create CHiME6
  local/data/generate_chime6_data.sh \
    --cmd "$train_cmd" \
    ${CHIME5_ROOT} \
    ${CHIME6_ROOT}
fi


if [ ${stage} -le 2 ] && ! contains $skip_stages 2; then
  if [ -d "$CHIME7Task1_ROOT" ]; then
    echo "${CHIME7Task1_ROOT} already exists, exiting"
    exit
  fi
  python local/data/generate_data.py -c $CHIME6_ROOT \
      -d $DIPCO_ROOT -m $MIXER6_ROOT -o $CHIME7Task1_ROOT
fi