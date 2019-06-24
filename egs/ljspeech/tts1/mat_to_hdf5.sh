#!/bin/bash

# (Katsuki Inoue)
# 

. ./path.sh
. ./cmd.sh

which python

# Begin configuration section.
nj=32
cmd=run.pl
compress=true
rfiletype=mat # mat or hdf5
wfiletype=hdf5 # mat or hdf5
write_utt2num_frames=true
# End configuration section.

model=sample_70-200 #train_no_dev_pytorch_taco2_r1_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_forward_ta128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs32_sd1
dir=eval # dev or eval

indir=exp/${model}/outputs_model.loss.best_th0.5_mlr0.0-10.0_denorm/${dir}
logdir=exp/${model}/hdf5/${dir}/log
outdir=exp/${model}/hdf5/${dir}

# make $fbankdir an absolute pathname.
indir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${indir} ${PWD})

## use "name" as part of name of the archive.
#name=$(basename ${indir})

mkdir -p ${logdir} || exit 1;
mkdir -p ${outdir} || exit 1;

if [ "${rfiletype}" == hdf5 ];then
    r_ext=h5
else
    r_ext=ark
fi

if [ "${wfiletype}" == hdf5 ];then
    w_ext=h5
else
    w_ext=ark
fi

if ${write_utt2num_frames}; then
  write_num_frames_opt="--write_num_frames=ark,t:${logdir}/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

#if [[ -e ${outdir}/feats.1.${w_ext} ]]; then
#    :
#else
#    echo "" > ${outdir}/feats.1.${w_ext}
#    echo "" > ${outdir}/feats.1.scp
#fi

${cmd} JOB=${nj} ${logdir}/mat_to_hdf5.JOB.log \
    convert_file_format.py \
        --compress=${compress} \
        ${write_num_frames_opt} \
        --rfiletype ${rfiletype} \
        --wfiletype ${wfiletype} \
        ark:${indir}/feats.${r_ext} \
        ark,scp:${outdir}/feats.${w_ext},${outdir}/feats.scp

#        ark:${indir}/feats.${JOB}.${r_ext} 
#        ark,scp:${outdir}/feats.${JOB}.${w_ext},${outdir}/feats.${JOB}.scp



