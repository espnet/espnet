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
./phonetize.sh all

# Run from stage 2
./run.sh \
    --train_set train_all_phn \
    --valid_set dev_all_phn \
    --test_sets "dev_all_phn eval1_all_phn" \
    --srctexts "data/train_all_phn/text" \
    --expdir "$expdir" \
    --stage 2 \
    --g2p none \
    --cleaner none \
    --use_xvector true \
    --min_wav_duration 0.38 \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/all/22k \
    --expdir exp/all/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_xvector_vits.yaml \
    --inference_model train.total_count.ave.pth
