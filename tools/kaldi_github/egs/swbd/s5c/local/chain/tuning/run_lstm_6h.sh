#!/bin/bash

# based on run_tdnn_6h.sh

# %WER 15.6 | 4459 42989 | 86.1 9.2 4.7 1.8 15.6 52.1 | exp/chain/lstm_6h_ld5_sp/decode_eval2000_sw1_fsh_fg/score_10_1.0/eval2000_hires.ctm.filt.sys
# %WER 10.3 | 1831 21395 | 90.9 6.1 3.0 1.3 10.3 44.7 | exp/chain/lstm_6h_ld5_sp/decode_eval2000_sw1_fsh_fg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 20.7 | 2628 21594 | 82.0 12.8 5.3 2.7 20.7 56.7 | exp/chain/lstm_6h_ld5_sp/decode_eval2000_sw1_fsh_fg/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys

# if right-tolerance was 10 (these are old results)
#---------------------------
# %WER 15.8 | 4459 42989 | 86.0 9.3 4.8 1.8 15.8 52.0 | exp/chain/lstm_6h_ld5_sp/decode_eval2000_sw1_fsh_fg/score_10_0.5/eval2000_hires.ctm.filt.sys
# %WER 10.6 | 1831 21395 | 90.6 6.2 3.2 1.2 10.6 45.2 | exp/chain/lstm_6h_ld5_sp/decode_eval2000_sw1_fsh_fg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 21.0 | 2628 21594 | 81.4 12.4 6.3 2.4 21.0 56.8 | exp/chain/lstm_6h_ld5_sp/decode_eval2000_sw1_fsh_fg/score_10_0.5/eval2000_hires.ctm.callhm.filt.sys

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/lstm_6h  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
chunk_width=150
chunk_left_context=40
chunk_right_context=0
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=5
# decode options
extra_left_context=
extra_right_context=
frames_per_chunk=

remove_egs=false
common_egs_dir=

affix=
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_2y_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
fi
repair_opts=${self_repair_scale:+" --self-repair-scale-nonlinearity $self_repair_scale "}
if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/lstm/make_configs.py  \
    $repair_opts \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --tree-dir $treedir \
    --splice-indexes="-2,-1,0,1,2 0 0" \
    --lstm-delay=" -3 -3 -3 " \
    --xent-regularize $xent_regularize \
    --include-log-softmax false \
    --num-lstm-layers 3 \
    --cell-dim 1024 \
    --hidden-dim 1024 \
    --recurrent-projection-dim 256 \
    --non-recurrent-projection-dim 256 \
    --label-delay $label_delay \
   $dir/configs || exit 1;

fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 15 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 250 --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
