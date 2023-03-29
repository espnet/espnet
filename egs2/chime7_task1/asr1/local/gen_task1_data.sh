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
gen_eval=0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./utils/parse_options.sh || exit 1


if [ -z "$chime5_root" ]; then
  skip_stages="1" # if chime5 undefined skip chime6 generation
fi

if [ ${stage} -le 0 ] && ! contains $skip_stages 0 ; then
  # download DiPCO
  if ! [ -d "${dipco_root}" ]; then
    mkdir -p ${dipco_root}
    if ! [ -f "${dipco_root}/dipco.tgz" ]; then
      wget https://s3.amazonaws.com/dipco/DiPCo.tgz -O ${dipco_root}/dipco.tgz
    fi

    if ! [ -d "${dipco_root}/audio" ]; then
      tar -xf ${dipco_root}/dipco.tgz -C ${dipco_root} --strip-components=1
    fi
  fi
fi


if [ ${stage} -le 1 ] && ! contains $skip_stages 1; then
  # from CHiME5 create CHiME6
  ./local/generate_chime6_data.sh --cmd "$cmd" \
    $chime5_root \
    $chime6_root
fi


if [ ${stage} -le 2 ] && ! contains $skip_stages 2; then
  python local/gen_task1_data.py -c $chime6_root \
      -d $dipco_root -m $mixer6_root -o $chime7_root --eval_opt $gen_eval
  if [ $gen_eval -le 0 ]; then
    # this creates UEM files for training portions of mixer6, where the UEM
    # file has start and end boundaries for the interview portion or the call
    # portion.
    mkdir -p $chime7_root/mixer6/uem/train_intv
    awk -F'[,_]' '(NR>1){print $4"_"$1"_"$2"_"$3"_"$4, 1, $7, $8}' \
    $mixer6_root/metadata/iv_components_final.csv | sort -u | cut -d'_' -f2- > $chime7_root/mixer6/uem/train_intv/all.uem
    # will produce ./chime7_dasr/mixer6/uem/train_intv/all.uem with contents:
    # 20090925_114220_HRM_110236 1 33.959 981.802
    # 20091006_131045_HRM_110236 1 59.169 1060.465
    # 20091110_121729_HRM_110236 1 27.828 1140.203

    mkdir -p $chime7_root/mixer6/uem/train_call
    awk -F'[,_]' '(NR>1){print $4"_"$1"_"$2"_"$3"_"$4, 1, $11, $12}' \
    $mixer6_root/metadata/iv_components_final.csv | sort -u | cut -d'_' -f2- > $chime7_root/mixer6/uem/train_call/all.uem
    # as before but for train_call portion.
  fi
fi
