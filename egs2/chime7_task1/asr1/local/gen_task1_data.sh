#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail
function contains ()  { [[ $1 =~ (^|[[:space:]])"$2"($|[[:space:]]) ]]; }

stage=1
cmd=run.pl
skip_stages="-1"
# this script creates and prepares the official Task 1 data from chime6, dipco and mixer6 datasets.
chime7_root=
chime5_root= # chime5 needs to be downloaded manually, but you can skip it if you have already CHiME-6
chime6_root= # will be created automatically from chime5
# but if you have it already you can use your existing one.
dipco_root=
mixer6_root=

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./utils/parse_options.sh || exit 1


if ! [ -d chime5_root ]; then
  skip_stages="1" # if chime5 undefined skip chime6 generation
fi

if [ ${stage} -le 0 ] && ! contains $skip_stages 0 ; then
  # download DiPCO
  if ! [ -d "${dipco_root}" ]; then
    mkdir -p ${dipco_root}
    if ! [ -f "${dipco_root}/dipco.tgz" ]; then
      wget https://s3.amazonaws.com/dipco/DiPCo.tgz -O ${dipco_root}/dipco.tgz
    fi

    if ! [-d "${dipco_root}/audio"]; then
      tar -xf ${dipco_root}/dipco.tgz -C ${dipco_root} --strip-components=1
    fi
  fi
fi


if [ ${stage} -le 1 ] && ! contains $skip_stages 1; then
  # from CHiME5 create CHiME6
  ./generate_chime6_data.sh --cmd "$cmd" \
    $chime5_root \
    $chime6_root
fi


if [ ${stage} -le 2 ] && ! contains $skip_stages 2; then
  if [ -d "${chime7_root}" ]; then
    echo "${chime7_root} already exists, exiting"
    exit
  fi
  python data/generate_data.py -c $chime6_root \
      -d $dipco_root -m $mixer6_root -o $chime7_root
fi
