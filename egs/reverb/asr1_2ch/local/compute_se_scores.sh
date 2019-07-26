#/bin/bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

# This script computes the dereverberation scores given in REVERB challenge
# Eg. local/compute_se_scores.sh --nch 8 /export/corpora5/REVERB_2014/REVERB ${PWD}/wav ${PWD}/local

. ./cmd.sh
. ./path.sh
set -e
set -u
set -o pipefail

cmd=run.pl
nch=8
enable_pesq=false

. utils/parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Wrong #arguments ($#, expected 5)"
   echo "Usage: local/compute_se.sh [options] <sim_scp> <real_scp> <reverb-data> <ref-scp> <pesq-directory>"
   echo "options"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --nch <nch>                              # nch of WPE to use for computing SE scores"
   echo "  --enable_pesq <enable_pesq>              # Boolean flag to enable PESQ"
   exit 1;
fi

enhancement_sim_scp=$1
enhancement_real_scp=$2
reverb_data=$3
ref_scp=$4
pesqdir=$5
root_dir=${PWD}
expdir=${PWD}/exp/compute_se_${nch}ch

if $enable_pesq; then
   compute_pesq=1
else
   compute_pesq=0
fi

pushd local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools
$cmd $expdir/compute_se_sim.log matlab -nodisplay -nosplash -r "addpath('SRMRToolbox'); score_SimData_scp('$reverb_data','$ref_scp','$root_dir','$enhancement_sim_scp','$pesqdir',$compute_pesq);exit"
$cmd $expdir/compute_se_real.log matlab -nodisplay -nosplash -r "addpath('SRMRToolbox'); score_RealData_scp('$root_dir','$enhancement_real_scp');exit"
popd

rm -rf $expdir/scores
mv local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/scores $expdir/

pushd local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools
$cmd $expdir/compute_se_sim.log matlab -nodisplay -nosplash -r "addpath('SRMRToolbox'); score_STOI_scp('$reverb_data','$ref_scp','$root_dir','$enhancement_sim_scp','$pesqdir',$compute_pesq);exit"
popd

echo "Calculating STOI and SDR"
for room in room1 room2 room3; do
    for dist in near far; do
	ref_scp=local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/et_${dist}_${room}_ref.scp
	est_scp=local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/et_${dist}_${room}_enh.scp
        eval_source_separation.sh --cmd "${train_cmd}" --nj 10 --bss-eval-images false --evaltypes "STOI" $ref_scp $est_scp exp/compute_se_${nch}ch/STOI/et_${dist}_${room}
    done
done