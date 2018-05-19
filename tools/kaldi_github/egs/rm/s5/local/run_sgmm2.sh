#!/bin/bash

# This is as run_sgmm.sh but using the "sgmm2" code, which uses "state-clustered tied mixtures"
# and the symmetric SGMM, and one or two other small changes (e.g. no updating of M for a few
# iterations.)

. ./cmd.sh
## SGMM on top of LDA+MLLT+SAT features.
if [ ! -f exp/ubm4a/final.ubm ] || [ ! data/train/feats.scp -nt exp/ubm4a/final.ubm ]; then
  steps/train_ubm.sh --silence-weight 0.5 --cmd "$train_cmd" 400 data/train data/lang exp/tri3b_ali exp/ubm4a || exit 1;
fi
          
steps/train_sgmm2.sh --cmd "$train_cmd" 5000 7000 data/train data/lang exp/tri3b_ali exp/ubm4a/final.ubm exp/sgmm2_4a || exit 1;

utils/mkgraph.sh data/lang exp/sgmm2_4a exp/sgmm2_4a/graph || exit 1;

steps/decode_sgmm2.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/sgmm2_4a/graph data/test exp/sgmm2_4a/decode || exit 1;

steps/decode_sgmm2.sh --use-fmllr true --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  --transform-dir exp/tri3b/decode  exp/sgmm2_4a/graph data/test exp/sgmm2_4a/decode_fmllr || exit 1;

 # Now we'll align the SGMM system to prepare for discriminative training.
 steps/align_sgmm2.sh --nj 8 --cmd "$train_cmd" --transform-dir exp/tri3b \
    --use-graphs true --use-gselect true data/train data/lang exp/sgmm2_4a exp/sgmm2_4a_ali || exit 1;
 steps/make_denlats_sgmm2.sh --nj 8 --sub-split 20 --cmd "$decode_cmd" --transform-dir exp/tri3b \
   data/train data/lang exp/sgmm2_4a_ali exp/sgmm2_4a_denlats
 steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri3b --boost 0.2 \
   data/train data/lang exp/sgmm2_4a_ali exp/sgmm2_4a_denlats exp/sgmm2_4a_mmi_b0.2 

 for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri3b/decode data/lang data/test exp/sgmm2_4a/decode exp/sgmm2_4a_mmi_b0.2/decode_it$iter &
 done  
(
 steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri3b --boost 0.2 --drop-frames true \
   data/train data/lang exp/sgmm2_4a_ali exp/sgmm2_4a_denlats exp/sgmm2_4a_mmi_b0.2_x 

 for iter in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
    --transform-dir exp/tri3b/decode data/lang data/test exp/sgmm2_4a/decode exp/sgmm2_4a_mmi_b0.2_x/decode_it$iter &
 done  
)
wait 
steps/decode_combine.sh data/test data/lang exp/tri1/decode exp/tri2a/decode exp/combine_1_2a/decode || exit 1;
steps/decode_combine.sh data/test data/lang exp/sgmm2_4a/decode exp/tri3b_mmi/decode exp/combine_sgmm2_4a_3b/decode || exit 1;
# combining the sgmm run and the best MMI+fMMI run.
steps/decode_combine.sh data/test data/lang exp/sgmm2_4a/decode exp/tri3b_fmmi_c/decode_it5 exp/combine_sgmm2_4a_3b_fmmic5/decode || exit 1;

steps/decode_combine.sh data/test data/lang exp/sgmm2_4a_mmi_b0.2/decode_it4 exp/tri3b_fmmi_c/decode_it5 exp/combine_sgmm2_4a_mmi_3b_fmmic5/decode || exit 1;

