#!/bin/bash

speaker_id=$1

if [ -z speaker_id ]; then
    echo "No speaker ID supplied. exiting..."
    exit 1
fi

expdir="exp/${speaker_id}"

# Use the previously trained tacotron2 model as the teacher
./run.sh \
    --ngpu 1 \
    --stage 7 \
    --train_set train_${speaker_id}_phn \
    --valid_set dev_${speaker_id}_phn \
    --test_sets "train_${speaker_id}_phn dev_${speaker_id}_phn eval1_${speaker_id}_phn" \
    --cleaner none \
    --g2p none \
    --train_config ./conf/tuning/train_tacotron2.yaml \
    --tts_exp exp_${speaker_id}/tts_train_tacotron2_raw_phn_none \
    --inference_args "--use_teacher_forcing true"

# Run fastspeech2 training
./run.sh \
    --ngpu 2 \
    --train_set train_${speaker_id}_phn \
    --valid_set dev_${speaker_id}_phn \
    --test_sets "dev_${speaker_id}_phn eval1_${speaker_id}_phn" \
    --srctexts "data/train_${speaker_id}_phn/text" \
    --expdir "$expdir" \
    --stage 5 \
    --g2p none \
    --cleaner none \
    --train_config ./conf/tuning/train_conformer_fastspeech2.yaml \
    --teacher_dumpdir exp_${speaker_id}/tts_train_tacotron2_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp_${speaker_id}/tts_train_tacotron2_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave/stats
