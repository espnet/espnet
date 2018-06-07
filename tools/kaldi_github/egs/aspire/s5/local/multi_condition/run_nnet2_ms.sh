#!/bin/bash

# This is the "multi-splice" version of the online-nnet2 training script.
# It's currently the best recipe for aspire.
# You'll notice that we splice over successively larger windows as we go deeper
# into the network. The temporal context used for training on reverberant data
# is larger than that used for other LVCSR recipes.

. ./cmd.sh


stage=1
train_stage=-10
use_gpu=true
dir=exp/nnet2_multicondition/nnet_ms_a

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512

  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    parallel_opts="$parallel_opts --config conf/queue_no_k20.conf --allow-k20 false"
    # that config is like the default config in the text of queue.pl, but adding the following lines.
    # default allow_k20=true
    # option allow_k20=true
    # option allow_k20=false -l 'hostname=!g01&!g02&!b06'
    # It's a workaround for an NVidia CUDA library bug for our currently installed version
    # of the CUDA toolkit, that only shows up on k20's
  fi
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

# do the common parts of the script.
local/multi_condition/run_nnet2_common.sh --stage $stage


if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/fisher_english_reverb-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  # The size of the system is kept rather smaller than the run_7a_960.sh system:
  # this is because we want it to be small enough that we could plausibly run it
  # in real-time.
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs 3 --num-jobs-initial 4 --num-jobs-final 22 \
    --num-hidden-layers 6 --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-10:-7:2:5" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_multicondition/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 4000 \
    --pnorm-output-dim 400 \
    --mix-up 12000 \
    --frames-per-eg 16 \
    --remove-egs false \
    data/train_rvb_hires data/lang exp/tri5a_rvb_ali $dir  || exit 1;
fi

if [ $stage -le 8 ]; then
  # dump iVectors for the testing data.
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data_dir}_hires exp/nnet2_multicondition/extractor exp/nnet2_multicondition/ivectors_${data_dir} || exit 1;
  done
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
   ( steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      --online-ivector-dir exp/nnet2_multicondition/ivectors_${data_dir} \
      exp/tri5a/graph data/${data_dir}_hires $dir/decode_${data_dir} || exit 1;
   ) &
  done
  wait;
fi


if [ $stage -le 10 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_multicondition/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 11 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
   ( steps/online/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" \
      --config conf/decode.config \
      exp/tri5a/graph data/${data_dir}_hires ${dir}_online/decode_${data_dir} || exit 1;
   ) &
  done
  wait;
fi

if [ $stage -le 12 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
   ( steps/online/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" \
      --config conf/decode.config \
      --per-utt true \
      exp/tri5a/graph data/${data_dir}_hires ${dir}_online/decode_${data_dir} || exit 1;
   ) &
  done
  wait;
fi

if [ $stage -le 13 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector (--online false)
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
   ( steps/online/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" \
      --config conf/decode.config \
      --per-utt true --online false \
      exp/tri5a/graph data/${data_dir}_hires ${dir}_online/decode_${data_dir} || exit 1;
   ) &
  done
fi

exit 0;
