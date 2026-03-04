#!/bin/bash

mkdir -p data/eval_phone
cp /mnt/ssd/jieun/datasets/저음질통화망/Eval/utt2wav_전화망_eval_koronly_rmvd.scp data/eval_phone/wav.scp
cp /mnt/ssd/jieun/datasets/저음질통화망/Eval/utt2txt_전화망_eval_koronly_rmvd.scp data/eval_phone/text
awk '{print $1, $1}' data/eval_phone/wav.scp > data/eval_phone/utt2spk
utils/utt2spk_to_spk2utt.pl data/eval_phone/utt2spk > data/eval_phone/spk2utt


utils/fix_data_dir.sh data/eval_phone