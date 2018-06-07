#!/bin/bash
# Copyright    2017  Hossein Hadian

# This script performs chain training in a flat-start manner
# and without building or using any context-dependency tree.
# It does not use ivecors or other forms of speaker adaptation
# except simple mean and variance normalization.
# It is called from run_e2e_phone.sh

# Note: this script is configured as phone-based, if you want
# to run it in character mode, you'll need to change _nosp
# to _char everywhere and also copy char_lm.fst instead
# of phone_lm.fst (in stage 1 below)

# local/chain/compare_wer.sh exp/chain/e2e_tdnn_1a
# System                e2e_tdnn_1a
#WER dev93 (tgpr)                9.70
#WER dev93 (tg)                  9.05
#WER dev93 (big-dict,tgpr)       7.20
#WER dev93 (big-dict,fg)         6.36
#WER eval92 (tgpr)               5.88
#WER eval92 (tg)                 5.32
#WER eval92 (big-dict,tgpr)      3.67
#WER eval92 (big-dict,fg)        3.05
# Final train prob        -0.0741
# Final valid prob        -0.0951
# Final train prob (xent)
# Final valid prob (xent)
# Num-params                 5562234

# steps/info/chain_dir_info.pl exp/chain/e2e_tdnn_1a
# exp/chain/e2e_tdnn_1a: num-iters=68 nj=2..5 num-params=5.6M dim=40->84 combine=-0.094->-0.094 logprob:train/valid[44,67,final]=(-0.083,-0.073,-0.072/-0.097,-0.095,-0.095)

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
affix=1a_dim450

# training options
num_epochs=4
num_jobs_initial=2
num_jobs_final=5
minibatch_size=150=128,64/300=100,64,32/600=50,32,16/1200=16,8
common_egs_dir=
l2_regularize=0.00005
dim=450
frames_per_iter=3000000
cmvn_opts="--norm-means=true --norm-vars=true"
train_set=train_si284_spe2e_hires
test_sets="test_dev93 test_eval92"

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

lang=data/lang_e2e
treedir=exp/chain/e2e_tree  # it's actually just a trivial tree (no tree building)
dir=exp/chain/e2e_tdnn_${affix}

if [ $stage -le 0 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_nosp $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 1 ]; then
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj 30 --cmd "$train_cmd" \
                                       --shared-phones true \
                                       data/$train_set $lang $treedir
  cp exp/chain/e2e_base/phone_lm.fst $treedir/
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.0025"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=40 name=input

  relu-batchnorm-layer name=tdnn1 input=Append(-1,0,1) dim=$dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$dim $opts
  relu-batchnorm-layer name=tdnn3 dim=$dim $opts
  relu-batchnorm-layer name=tdnn4 input=Append(-1,0,1) dim=$dim $opts
  relu-batchnorm-layer name=tdnn5 dim=$dim $opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$dim $opts
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$dim $opts
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=$dim $opts

  relu-batchnorm-layer name=prefinal-chain dim=$dim target-rms=0.5 $opts
  output-layer name=output include-log-softmax=true dim=$num_targets $output_opts

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "" \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 4 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_nosp_test_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_nosp_test_tgpr \
    $dir $treedir/graph_tgpr || exit 1;

  utils/lang/check_phones_compatible.sh \
    data/lang_nosp_test_bd_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_nosp_test_bd_tgpr \
    $dir $treedir/graph_bd_tgpr || exit 1;
fi

if [ $stage -le 5 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          $treedir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_nosp_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

echo "Done. Date: $(date). Results:"
local/chain/compare_wer.sh $dir
