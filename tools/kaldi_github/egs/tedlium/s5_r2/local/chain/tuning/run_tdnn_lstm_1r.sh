#!/bin/bash

# 1r is as 1e, but changing update-period of natural gradient from 4 to 1,
# Not helpful.

# local/chain/compare_wer_general.sh --looped exp/chain_cleaned/tdnn_lstm1{e,r}_sp_bi
# local/chain/compare_wer_general.sh --looped exp/chain_cleaned/tdnn_lstm1e_sp_bi exp/chain_cleaned/tdnn_lstm1r_sp_bi
# System                tdnn_lstm1e_sp_bi tdnn_lstm1r_sp_bi
# WER on dev(orig)            9.0       9.0
#         [looped:]           9.0       9.1
# WER on dev(rescored)        8.4       8.5
#         [looped:]           8.4       8.6
# WER on test(orig)           8.8       9.1
#         [looped:]           8.8       9.0
# WER on test(rescored)       8.4       8.4
#         [looped:]           8.3       8.5
# Final train prob        -0.0648   -0.0642
# Final valid prob        -0.0827   -0.0838
# Final train prob (xent)   -0.8372   -0.8319
# Final valid prob (xent)   -0.9497   -0.9635

# 1e is as 1b, but reducing decay-time from 40 to 20.

# 1d is as 1b, but adding decay-time=40 to the fast-lstmp-layers.  note: it
# uses egs from 1b, remember to remove that before I commit.

# steps/info/chain_dir_info.pl exp/chain_cleaned/tdnn_lstm1a_sp_bi
# exp/chain_cleaned/tdnn_lstm1a_sp_bi: num-iters=253 nj=2..12 num-params=9.5M dim=40+100->3607 combine=-0.07->-0.07 xent:train/valid[167,252,final]=(-0.960,-0.859,-0.852/-1.05,-0.999,-0.997) logprob:train/valid[167,252,final]=(-0.076,-0.064,-0.062/-0.099,-0.092,-0.091)

# This is as run_lstm1e.sh except adding TDNN layers in between; also comparing below
# with run_lstm1d.sh which had a larger non-recurrent-projection-dim and which had
# better results.  Note: these results are not with the updated LM (the LM data-prep
# for this setup was changed in Nov 2016 but this was with an older directory).
#
# local/chain/compare_wer_general.sh exp/chain_cleaned/lstm1d_sp_bi exp/chain_cleaned/lstm1e_sp_bi exp/chain_cleaned/tdnn_lstm1a_sp_bi
# System                lstm1d_sp_bi lstm1e_sp_bi tdnn_lstm1a_sp_bi
# WER on dev(orig)          10.3      10.7       9.7
# WER on dev(rescored)       9.8      10.1       9.3
# WER on test(orig)           9.7       9.8       9.1
# WER on test(rescored)       9.2       9.4       8.7
# Final train prob        -0.0812   -0.0862   -0.0625
# Final valid prob        -0.1049   -0.1047   -0.0910
# Final train prob (xent)   -1.1334   -1.1763   -0.8518
# Final valid prob (xent)   -1.2263   -1.2427   -0.9972

## how you run this (note: this assumes that the run_tdnn_lstm.sh soft link points here;
## otherwise call it directly in its location).
# by default, with cleanup:
# local/chain/run_tdnn_lstm.sh

# without cleanup:
# local/chain/run_tdnn_lstm.sh  --train-set train --gmm tri3 --nnet3-affix "" &

# note, if you have already run one of the non-chain nnet3 systems
# (e.g. local/nnet3/run_tdnn.sh), you may want to run with --stage 14.

# run_tdnn_lstm_1a.sh was modified from run_lstm_1e.sh, which is a fairly
# standard, LSTM, except that some TDNN layers were added in between the
# LSTM layers.  I was looking at egs/ami/s5b/local/chain/tuning/run_tdnn_lstm_1i.sh, but
# this isn't exactly copied from there.


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
decode_nj=30
min_seg_len=1.55
label_delay=5
xent_regularize=0.1
train_set=train_cleaned
gmm=tri3_cleaned  # the gmm for the target data
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
# training options
chunk_left_context=40
chunk_right_context=0
chunk_left_context_initial=0
chunk_right_context_final=0
frames_per_chunk=140,100,160
# decode options
frames_per_chunk_primary=$(echo $frames_per_chunk | cut -d, -f1)
extra_left_context=50
extra_right_context=0
extra_left_context_initial=0
extra_right_context_final=0


# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_lstm_affix=1r  #affix for TDNN-LSTM directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=exp/chain_cleaned/tdnn_lstm1b_sp_bi/egs  # you can set this to use previously dumped egs.
remove_egs=true

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                             --nnet3-affix "$nnet3_affix"


gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
tree_dir=exp/chain${nnet3_affix}/tree_bi${tree_affix}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats
dir=exp/chain${nnet3_affix}/tdnn_lstm${tdnn_lstm_affix}_sp_bi
train_data_dir=data/${train_set}_sp_hires_comb
lores_train_data_dir=data/${train_set}_sp_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 14 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 15 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 16 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi


if [ $stage -le 17 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  tdnn_opts='ng-affine-options="update-period=1"'
  lstmp_opts='ng-affine-options="update-period=1" decay-time=20'
  output_opts='max-change=1.5 ng-affine-options="update-period=1"'

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=512 $tdnn_opts
  relu-renorm-layer name=tdnn2 dim=512 input=Append(-1,0,1) $tdnn_opts
  fast-lstmp-layer name=lstm1 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstmp_opts
  relu-renorm-layer name=tdnn3 dim=512 input=Append(-3,0,3) $tdnn_opts
  relu-renorm-layer name=tdnn4 dim=512 input=Append(-3,0,3) $tdnn_opts
  fast-lstmp-layer name=lstm2 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstmp_opts
  relu-renorm-layer name=tdnn5 dim=512 input=Append(-3,0,3) $tdnn_opts
  relu-renorm-layer name=tdnn6 dim=512 input=Append(-3,0,3) $tdnn_opts
  fast-lstmp-layer name=lstm3 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstmp_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 18 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width "$frames_per_chunk" \
    --egs.chunk-left-context "$chunk_left_context" \
    --egs.chunk-right-context "$chunk_right_context" \
    --egs.chunk-left-context-initial "$chunk_left_context_initial" \
    --egs.chunk-right-context-final "$chunk_right_context_final" \
    --trainer.num-chunk-per-minibatch 128,64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
    --trainer.deriv-truncate-margin 10 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --cleanup.remove-egs "$remove_egs" \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi



if [ $stage -le 19 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
fi

if [ $stage -le 20 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --extra-left-context-initial $extra_left_context_initial \
          --extra-right-context-final $extra_right_context_final \
          --frames-per-chunk "$frames_per_chunk_primary" \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph data/${dset}_hires $dir/decode_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


if [ $stage -le 21 ]; then
  # 'looped' decoding.  we didn't write a -parallel version of this program yet,
  # so it will take a bit longer as the --num-threads option is not supported.
  # we just hardcode the --frames-per-chunk option as it doesn't have to
  # match any value used in training, and it won't affect the results (unlike
  # regular decoding).
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
      steps/nnet3/decode_looped.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context-initial $extra_left_context_initial \
          --frames-per-chunk 30 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph data/${dset}_hires $dir/decode_looped_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}/decode_looped_${dset} ${dir}/decode_looped_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


exit 0
