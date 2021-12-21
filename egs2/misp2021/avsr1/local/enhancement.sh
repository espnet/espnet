#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Zhaoxu Nian, Hang Chen, Yen-Ju Lu)
# Apache 2.0

# use nara-wpe and BeamformIt to enhance multichannel data

set -e -o pipefail

# configs
stage=0
nj=6
cmd=run.pl

. utils/parse_options.sh

. ./path.sh || exit 1;

pip show -f nara_wpe >/dev/null || pip install nara_wpe

if [ -z $BEAMFORMIT ] ; then
  export BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt
fi
export PATH=${PATH}:$BEAMFORMIT
! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/kaldi/tools; extras/install_beamformit.sh; cd -;'" && exit 1

if [ $# != 2 ]; then
 echo "Usage: $0 <corpus-data-dir> <enhancement-dir>"
 echo " $0 /path/misp2021 /path/wpe_output"
 exit 1;
fi

data_root=$1
out_root=$2

echo "start speech enhancement"
# wpe
if [ $stage -le 0 ]; then
  echo "start wpe"
  python local/find_wav.py -nj $nj $data_root $out_root/log wpe Far
  for n in `seq $nj`; do
    cat <<-EOF > $out_root/log/wpe.$n.sh
    python local/run_wpe.py $out_root/log/wpe.$n.scp $data_root $out_root
EOF
  done
  chmod a+x $out_root/log/wpe.*.sh
  $cmd JOB=1:$nj $out_root/log/wpe.JOB.log $out_root/log/wpe.JOB.sh
  echo "finish wpe"
fi

# BeamformIt
if [ $stage -le 1 ]; then
  echo "start beamformit"
  python local/find_wav.py $PWD/$out_root $out_root/log beamformit Far
  python local/run_beamformit.py $BEAMFORMIT/BeamformIt conf/beamformit.cfg / $out_root/log/beamformit.scp $out_root
  echo "end beamformit"
fi
echo "end speech enhancement"
