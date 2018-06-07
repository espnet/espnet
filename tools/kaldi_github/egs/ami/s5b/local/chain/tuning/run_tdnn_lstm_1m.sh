#!/bin/bash

# This (1m.sh) is the same as 1j but with per-frame dropout on LSTM layer
# It is a fast LSTM with per-frame dropout on [i, f, o] gates of the LSTM,
# the dropout-adding place is "place4" in paper : http://www.danielpovey.com/files/2017_interspeech_dropout.pdf.
# We have tried both 4-epoch and 5-epoch training.

### IHM
# Results with flags : --mic ihm  --train-set train_cleaned  --gmm tri3_cleaned \
#System            tdnn_lstm1j_sp_bi_ld5 tdnn_lstm1m_sp_bi_ld5
#WER on dev        20.8      19.9
#WER on eval        20.3      19.3
#Final train prob     -0.0439145 -0.0653269
#Final valid prob       -0.10673 -0.0998743
#Final train prob (xent)     -0.683776 -0.884698
#Final valid prob (xent)      -1.05254  -1.09002

# steps/info/chain_dir_info.pl exp/ihm/chain_cleaned/tdnn_lstm1j_sp_bi_ld5/ exp/ihm/chain_cleaned/tdnn_lstm1m_sp_bi_ld5/
# exp/ihm/chain_cleaned/tdnn_lstm1j_sp_bi_ld5: num-iters=89 nj=2..12 num-params=43.4M dim=40+100->3765 combine=-0.063->-0.058 xent:train/valid[58,88,final]=(-0.888,-0.695,-0.684/-1.12,-1.06,-1.05) logprob:train/valid[58,88,final]=(-0.065,-0.045,-0.044/-0.105,-0.107,-0.107)
# exp/ihm/chain_cleaned/tdnn_lstm1m_sp_bi_ld5: num-iters=89 nj=2..12 num-params=43.4M dim=40+100->3765 combine=-0.092->-0.080 xent:train/valid[58,88,final]=(-3.12,-1.09,-0.885/-3.20,-1.27,-1.09) logprob:train/valid[58,88,final]=(-0.164,-0.072,-0.065/-0.181,-0.103,-0.100)

# Results with flags for (1m.sh) : --num-epochs 5 --tlstm-affix 1m_5epoch --mic ihm  --train-set train_cleaned  --gmm tri3_cleaned \
# Results with flags for (1j.sh) : --num-epochs 5 --tlstm-affix 1j_5epoch --mic ihm  --train-set train_cleaned  --gmm tri3_cleaned \
#System            tdnn_lstm1j_5epoch_sp_bi_ld5 tdnn_lstm1m_5epoch_sp_bi_ld5
#WER on dev        21.1      19.9
#WER on eval        20.9      19.8
#Final train prob     -0.0365079 -0.057024
#Final valid prob      -0.112709-0.0992725
#inal train prob (xent)     -0.601602 -0.800653
#Final valid prob (xent)      -1.03241  -1.04748

# ./steps/info/chain_dir_info.pl exp/ihm/chain_cleaned/tdnn_lstm1j_5epoch_sp_bi_ld5/ exp/ihm/chain_cleaned/tdnn_lstm1m_5epoch_sp_bi_ld5/
# exp/ihm/chain_cleaned/tdnn_lstm1j_5epoch_sp_bi_ld5/: num-iters=111 nj=2..12 num-params=43.4M dim=40+100->3765 combine=-0.053->-0.049 xent:train/valid[73,110,final]=(-0.813,-0.615,-0.602/-1.08,-1.04,-1.03) logprob:train/valid[73,110,final]=(-0.057,-0.038,-0.037/-0.106,-0.113,-0.113)
# exp/ihm/chain_cleaned/tdnn_lstm1m_5epoch_sp_bi_ld5/: num-iters=111 nj=2..12 num-params=43.4M dim=40+100->3765 combine=-0.080->-0.072 xent:train/valid[73,110,final]=(-3.15,-0.985,-0.801/-3.26,-1.21,-1.05) logprob:train/valid[73,110,final]=(-0.161,-0.062,-0.057/-0.183,-0.102,-0.099)

#### SDM
# Results with flags : --mic sdm1 --use-ihm-ali true --train-set train_cleaned  --gmm tri3_cleaned \
#System            tdnn_lstm1j_sp_bi_ihmali_ld5 tdnn_lstm1m_sp_bi_ihmali_ld5
#WER on dev        36.9      36.4
#WER on eval        40.5      39.9
#Final train prob      -0.108141 -0.148861
#Final valid prob      -0.257468 -0.240962
#Final train prob (xent)      -1.38179  -1.70258
#Final valid prob (xent)      -2.13095  -2.12803

# ./steps/info/chain_dir_info.pl exp/sdm1/chain_cleaned/tdnn_lstm1j_sp_bi_ihmali_ld5/ exp/sdm1/chain_cleaned/tdnn_lstm1m_sp_bi_ihmali_ld5/
# exp/sdm1/chain_cleaned/tdnn_lstm1j_sp_bi_ihmali_ld5/: num-iters=87 nj=2..12 num-params=43.4M dim=40+100->3741 combine=-0.138->-0.128 xent:train/valid[57,86,final]=(-1.71,-1.39,-1.38/-2.18,-2.14,-2.13) logprob:train/valid[57,86,final]=(-0.150,-0.110,-0.108/-0.251,-0.260,-0.257)
# exp/sdm1/chain_cleaned/tdnn_lstm1m_sp_bi_ihmali_ld5/: num-iters=87 nj=2..12 num-params=43.4M dim=40+100->3741 combine=-0.187->-0.170 xent:train/valid[57,86,final]=(-3.74,-1.90,-1.70/-3.88,-2.28,-2.13) logprob:train/valid[57,86,final]=(-0.286,-0.158,-0.149/-0.336,-0.245,-0.241)

# Results with flags for (1m.sh) : --num-epochs 5 --tlstm-affix 1m_5epoch --mic sdm1 --use-ihm-ali true --train-set train_cleaned  --gmm tri3_cleaned\
# Results with flags for (1j.sh) : --num-epochs 5 --tlstm-affix 1j_5epoch --mic sdm1 --use-ihm-ali true --train-set train_cleaned  --gmm tri3_cleaned\
#System            tdnn_lstm1j_5epoch_sp_bi_ihmali_ld5 tdnn_lstm1m_5epoch_sp_bi_ihmali_ld5
#WER on dev        37.4      36.0
#WER on eval        40.7      39.6
#Final train prob     -0.0879063 -0.133092
#Final valid prob      -0.270953 -0.243246
#Final train prob (xent)      -1.20822  -1.56293
#Final valid prob (xent)       -2.1425  -2.07265

# ./steps/info/chain_dir_info.pl exp/sdm1/chain_cleaned/tdnn_lstm1j_5epoch_sp_bi_ihmali_ld5/ exp/sdm1/chain_cleaned/tdnn_lstm1m_5epoch_sp_bi_ihmali_ld5/
# exp/sdm1/chain_cleaned/tdnn_lstm1j_5epoch_sp_bi_ihmali_ld5/: num-iters=109 nj=2..12 num-params=43.4M dim=40+100->3741 combine=-0.115->-0.107 xent:train/valid[71,108,final]=(-1.56,-1.22,-1.21/-2.16,-2.16,-2.14) logprob:train/valid[71,108,final]=(-0.131,-0.090,-0.088/-0.256,-0.273,-0.271)
# exp/sdm1/chain_cleaned/tdnn_lstm1m_5epoch_sp_bi_ihmali_ld5/: num-iters=109 nj=2..12 num-params=43.4M dim=40+100->3741 combine=-0.167->-0.153 xent:train/valid[71,108,final]=(-3.69,-1.71,-1.56/-3.84,-2.20,-2.07) logprob:train/valid[71,108,final]=(-0.279,-0.140,-0.133/-0.329,-0.247,-0.243)


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
min_seg_len=1.55
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # the gmm for the target data
ihm_gmm=tri3  # the gmm for the IHM system (if --use-ihm-ali true).
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
dropout_schedule='0,0@0.20,0.3@0.50,0' # dropout schedule controls the dropout
                                       # proportion for each training iteration.
num_epochs=4

chunk_width=150
chunk_left_context=40
chunk_right_context=0
label_delay=5
# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tlstm_affix=1m  #affix for TDNN-LSTM directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.


# decode options
extra_left_context=50
frames_per_chunk=

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
                                  --mic $mic \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"

# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --min-seg-len $min_seg_len \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set

if $use_ihm_ali; then
  gmm_dir=exp/ihm/${ihm_gmm}
  ali_dir=exp/${mic}/${ihm_gmm}_ali_${train_set}_sp_comb_ihmdata
  lores_train_data_dir=data/$mic/${train_set}_ihmdata_sp_comb
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}_ihmdata
  lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats_ihmdata
  dir=exp/$mic/chain${nnet3_affix}/tdnn_lstm${tlstm_affix}_sp_bi_ihmali
  # note: the distinction between when we use the 'ihmdata' suffix versus
  # 'ihmali' is pretty arbitrary.
else
  gmm_dir=exp/${mic}/$gmm
  ali_dir=exp/${mic}/${gmm}_ali_${train_set}_sp_comb
  lores_train_data_dir=data/$mic/${train_set}_sp_comb
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}
  lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats
  dir=exp/$mic/chain${nnet3_affix}/tdnn_lstm${tlstm_affix}_sp_bi
fi

if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi

train_data_dir=data/$mic/${train_set}_sp_hires_comb
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7


for f in $gmm_dir/final.mdl $lores_train_data_dir/feats.scp \
   $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 11 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning perturbed, short-segment-combined ${maybe_ihm}data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
     ${lores_train_data_dir} data/lang $gmm_dir $ali_dir
fi

[ ! -f $ali_dir/ali.1.gz ] && echo  "$0: expected $ali_dir/ali.1.gz to exist" && exit 1

if [ $stage -le 12 ]; then
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

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
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
      --cmd "$train_cmd" 4200 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

xent_regularize=0.1

if [ $stage -le 15 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=20 dropout-proportion=0.0"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=1024
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=lstm1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=lstm2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn8 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn9 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=lstm3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
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
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


graph_dir=$dir/graph_${LM}
if [ $stage -le 17 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 18 ]; then
  rm $dir/.error 2>/dev/null || true

  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;

  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --extra-left-context $extra_left_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --online-ivector-dir exp/$mic/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
