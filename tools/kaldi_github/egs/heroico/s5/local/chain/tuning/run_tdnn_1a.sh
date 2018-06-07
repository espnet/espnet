#!/bin/bash
# 1e
# lower number of epochs to 7 from 10 (avoid overfitting?)

# compare with 1d
# ./local/chain/compare_wer.sh exp/chain/tdnn1d_sp exp/chain/tdnn1e_sp
# System                tdnn1d_sp tdnn1e_sp
#WER devtest       52.78     52.21
#WER native       55.32     53.43
nonnative     64.35     61.03
# test       60.28     57.70
# Final train prob        -0.0229   -0.0250
# Final valid prob        -0.0683   -0.0678
# Final train prob (xent)   -0.7525   -0.7887
# Final valid prob (xent)   -1.0296   -1.0419

# info
#exp/chain/tdnn1e_sp:
# num-iters=105
# nj=1..1
# num-params=6.6M
# dim=40+100->1392
# combine=-0.036->-0.033
# xent:train/valid[69,104,final]=(-1.20,-0.917,-0.789/-1.35,-1.16,-1.04)
# logprob:train/valid[69,104,final]=(-0.049,-0.030,-0.025/-0.082,-0.075,-0.068)

# Word Error Rates on folds
%WER 61.03 [ 5624 / 9215, 630 ins, 727 del, 4267 sub ] exp/chain/tdnn1e_sp/decode_nonnative/wer_8_1.0
%WER 57.70 [ 9644 / 16713, 1249 ins, 1040 del, 7355 sub ] exp/chain/tdnn1e_sp/decode_test/wer_7_1.0
%WER 53.43 [ 4006 / 7498, 558 ins, 408 del, 3040 sub ] exp/chain/tdnn1e_sp/decode_native/wer_7_1.0
%WER 52.21 [ 3994 / 7650, 585 ins, 456 del, 2953 sub ] exp/chain/tdnn1e_sp/decode_devtest/wer_9_1.0

# | fold | 1a | 1b | 1c | 1d | 1e |
#| devtest | 54.46 | 54.20 | 54.16 | 52.78 | 52.21 |
#| native |  62.14 | 62.32 | 61.70 | 55.32 | 53.43 |
#| nonnative | 70.58 | 71.20 | 71.68 | 64.35 | 61.03 |
#| test | 66.85 | 67.21 | 67.25 | 60.28 | 57.70 |

# this script came from the mini librispeech recipe
# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train
test_sets="native nonnative devtest test"
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1e   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

num_leaves=3500

# training options
# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.

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
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --cmd "$train_cmd" \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    $num_leaves \
    ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 13 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.0025"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $opts dim=512
  relu-batchnorm-layer name=tdnn2 $opts dim=512 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 $opts dim=512
  relu-batchnorm-layer name=tdnn4 $opts dim=512 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn5 $opts dim=512
  relu-batchnorm-layer name=tdnn6 $opts dim=512 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn7 $opts dim=512 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn8 $opts dim=512 input=Append(-6,-3,0)

  # adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain $opts dim=512 target-rms=0.5
  output-layer name=output $output_opts include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent $opts input=tdnn8 dim=512 target-rms=0.5
  output-layer name=output-xent $output_opts dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  steps/nnet3/chain/train.py \
    --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=7 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=256,128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_test \
    $tree_dir \
    $tree_dir/graph || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/nnet3/decode.sh \
      --acwt 1.0 \
      --post-decode-acwt 10.0 \
      --extra-left-context $chunk_left_context \
      --extra-right-context $chunk_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nspk \
      --cmd "$decode_cmd" \
      --num-threads 4 \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
      $tree_dir/graph \
      data/${data}_hires \
      ${dir}/decode_${data} || exit 1;
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang \
    exp/nnet3${nnet3_affix}/extractor \
    ${dir} \
    ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    # note: we just give it "data/${data}" as it only uses the wav.scp, the
    # feature type does not matter.
    steps/online/nnet3/decode.sh \
      --acwt 1.0 \
      --post-decode-acwt 10.0 \
      --nj $nspk \
      --cmd "$decode_cmd" \
      $tree_dir/graph \
      data/${data} \
      ${dir}_online/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;

# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# End:
