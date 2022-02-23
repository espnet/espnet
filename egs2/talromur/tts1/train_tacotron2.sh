#!/bin/bash

speaker_id=$1

if [ -z $speaker_id ]; then
    echo "Speaker id was not provided. Please provide a speaker id from the available ids: [a, b, c, d, e, f, g, h]"
    exit 1
fi

expdir="exp/${speaker_id}"

train_set=train_${speaker_id}
valid_set=dev_${speaker_id}
eval_set=eval1_${speaker_id}
test_sets="${valid_set} ${eval_set}"

# # Prep data directory
./run.sh --stage 1 --stop-stage 1 \
    --train_set "$train_set" \
    --valid_set "$valid_set" \
    --test_sets "$test_sets" \
    --srctexts "data/train_${speaker_id}/text" \
    --expdir "$expdir" \
    --local_data_opts "$speaker_id"

# # Since espeak is super slow, dump phonemized text at first
./phonetize.sh $speaker_id

# Run from stage 2
./run.sh \
    --train_set train_${speaker_id}_phn \
    --valid_set dev_${speaker_id}_phn \
    --test_sets "dev_${speaker_id}_phn eval1_${speaker_id}_phn" \
    --srctexts "data/train_${speaker_id}_phn/text" \
    --expdir "$expdir" \
    --stage 2 \
    --g2p none \
    --cleaner none \
    --ngpu 2 \
    --train_config ./conf/tuning/train_tacotron2.yaml