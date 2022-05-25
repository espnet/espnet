#!/bin/bash

expdir="exp/all"

train_set=train_all
valid_set=dev_all
eval_set=eval1_all
test_sets="${valid_set} ${eval_set}"

# # Prep data directory
./run.sh --stage 1 --stop-stage 1 \
    --train_set "$train_set" \
    --valid_set "$valid_set" \
    --test_sets "$test_sets" \
    --srctexts "data/train_all/text" \
    --expdir "$expdir" \
    --local_data_opts "all"

# # Since ice-g2p phonetization is very slow, dump phonemized text at first
./local/phonetize.sh all

# Run from stage 2
./run.sh \
    --train_set "${train_set}_phn" \
    --valid_set "${valid_set}_phn" \
    --test_sets "${valid_set}_phn ${eval_set}_phn" \
    --srctexts "data/${train_set}_phn/text" \
    --expdir "$expdir" \
    --stage 2 \
    --g2p none \
    --cleaner none \
    --use_xvector true \
    --ngpu 4 \
    --expdir "$expdir" \
    --train_config ./conf/tuning/train_xvector_tacotron2.yaml \
    --inference_model dev.total_count.ave.pth