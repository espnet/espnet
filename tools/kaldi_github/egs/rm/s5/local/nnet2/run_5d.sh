#!/bin/bash


# This script demonstrates discriminative training of p-norm neural nets.
# It's on top of run_4d_gpu.sh which uses adapted 40-dimensional features.
# This version of the script uses GPUs.  We distinguish it by putting "_gpu"
# at the end of the directory name.


use_gpu=true
stage=0
transform_dir=exp/tri3b_ali

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


[ ! -f $transform_dir/num_jobs ] && \
  echo "Expected $transform_dir/num_jobs to exist" && exit 1;
nj_orig=$(cat $transform_dir/num_jobs)


# The queue options in this script are for the CLSP network, and might not work
# for you.

if $use_gpu; then
  . ./cmd.sh
  . ./path.sh
  ! cuda-compiled && cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  align_gpu_opts="--gpu 1"
  use_gpu_flag="--use-gpu yes"
  train_parallel_opts="--gpu 1"
  train_num_threads=1
  srcdir=exp/nnet4d_gpu
  dir=exp/nnet5d_mpe_gpu
  nj=$nj_orig
else
  align_gpu_opts=
  use_gpu_flag="--use-gpu no"
  train_parallel_opts="--num-threads 6"
  train_num_threads=6
  srcdir=exp/nnet4d
  dir=exp/nnet5d_mpe
  if [ "$decode_cmd" != "run.pl" ]; then
    nj=$[$nj_orig*5]; # use more jobs, or it will be slow in the alignment
                      # phase.  But if we are just running everything on
                      # one machine this won't help us
  else
    nj=$nj_orig
  fi
fi

if [ ! -f $srcdir/final.mdl ]; then
  echo "$0: expected $srcdir/final.mdl to exist."
  exit 1;
fi

# The denominator lattice creation currently doesn't use GPUs; that would be
# wasteful since the lattice determinization and graph search use up a fair
# amount of CPU, and we'd be idling the GPU much of the time.

# We specify 1G each for --mem, which is per thread... it
# will likely be less than the default.  Increase the beam relative to the
# defaults; this is just for this RM setup, where the default beams will likely
# generate very thin lattices.

#  Note: the transform-dir is important to
# specify, since this system is on top of fMLLR features.


if [ $stage -le 0 ]; then
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
    --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "--num-threads 6" \
    --beam 20.0 --lattice-beam 10.0 \
    --transform-dir $transform_dir \
    data/train data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 1 ]; then
  steps/nnet2/align.sh  --cmd "$decode_cmd $align_gpu_opts" $use_gpu_flag \
    --transform-dir $transform_dir \
    --nj $nj data/train data/lang $srcdir ${srcdir}_ali
fi
if [ $stage -le 2 ]; then
  steps/nnet2/train_discriminative.sh --cmd "$decode_cmd" \
    --num-jobs-nnet 2 --transform-dir $transform_dir \
    --num-threads "$train_num_threads" --parallel-opts "$train_parallel_opts" data/train data/lang \
    ${srcdir}_ali ${srcdir}_denlats $srcdir/final.mdl $dir
fi
if [ $stage -le 3 ]; then
  for epoch in 1 2 3 4; do
    steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch$epoch \
      --transform-dir exp/tri3b/decode \
      exp/tri3b/graph data/test $dir/decode_epoch$epoch  &

    steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --iter epoch$epoch \
      --transform-dir exp/tri3b/decode_ug \
      exp/tri3b/graph_ug data/test $dir/decode_ug_epoch$epoch &
  done
  wait
fi

exit 0;



# The following is some test commands that I ran in order to verify that
# the neural-net splitting and excising code was working as intended.

# (
# acoustic_scale=0.1
# for criterion in smbr mmi mpfe; do
#   for drop_frames in true false; do
#     nnet-get-egs-discriminative  --drop-frames=$drop_frames  --criterion=$criterion --excise=true exp/tri5c_mpe/0.mdl 'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/train/split8/1/utt2spk scp:data/train/split8/1/cmvn.scp "scp:head -n 40 data/train/split8/1/feats.scp|" ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/tri5c_mpe/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/train/split8/1/utt2spk ark:$transform_dir/trans.1 ark:- ark:- |' 'ark,s,cs:gunzip -c exp/${dir}_ali/ali.1.gz |' 'ark,s,cs:gunzip -c exp/${dir}_denlats/lat.1.gz|' "ark:|nnet-combine-egs-discriminative ark:- ark:1.egs"

#     nnet-get-egs-discriminative --drop-frames=$drop_frames --criterion=$criterion --split=false --excise=false exp/tri5c_mpe/0.mdl 'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/train/split8/1/utt2spk scp:data/train/split8/1/cmvn.scp "scp:head -n 40 data/train/split8/1/feats.scp|" ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/tri5c_mpe/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/train/split8/1/utt2spk ark:$transform_dir/trans.1 ark:- ark:- |' 'ark,s,cs:gunzip -c exp/${dir}_ali/ali.1.gz |' 'ark,s,cs:gunzip -c exp/${dir}_denlats/lat.1.gz|' ark:2.egs

#    nnet-compare-hash-discriminative --acoustic-scale=$acoustic_scale --drop-frames=$drop_frames --criterion=$criterion $dir/final.mdl ark:1.egs ark:2.egs || exit 1;
#  done
# done
# )
