#!/usr/bin/env bash
set -euo pipefail

src_lang=eng
tgt_lang=deu
task=st
max_train_samples=0
max_valid_samples=0
max_test_samples=0

. ./db.sh
. ./path.sh
. utils/parse_options.sh

mkdir -p "${MULTIMED_ST}"

python local/prepare_multimed_st.py \
  --src_lang "${src_lang}" \
  --tgt_lang "${tgt_lang}" \
  --task "${task}" \
  --out_root "${MULTIMED_ST}/${src_lang}_${tgt_lang}_${task}" \
  --max_train_samples "${max_train_samples}" \
  --max_valid_samples "${max_valid_samples}" \
  --max_test_samples "${max_test_samples}"

mkdir -p data
rm -rf data/train data/valid data/test
cp -r "${MULTIMED_ST}/${src_lang}_${tgt_lang}_${task}/data/train" data/train
cp -r "${MULTIMED_ST}/${src_lang}_${tgt_lang}_${task}/data/valid" data/valid
cp -r "${MULTIMED_ST}/${src_lang}_${tgt_lang}_${task}/data/test" data/test

for dset in train valid test; do
  utils/fix_data_dir.sh --utt_extra_files "text.prev text.ctc" data/${dset}
  utils/validate_data_dir.sh --no-feats data/${dset}
done
