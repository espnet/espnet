stage=1
asr_exp=exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/k2_output/
if [ $stage -le 0 ]; then
  # To make sure code, model and data are available,
  # stage > 0 assumes you run following statements successfully,
  # reference:
  # https://zenodo.org/record/4604066#.YPUlxOgzZPY
  git checkout 70d16d210cdce28e71b7892a0ec96eaaa6474d64
  pip install -e .
  cd egs2/librispeech/asr1
  ./run.sh --skip_data_prep false --skip_train true --download_model kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave

fi
if [ $stage -le 1 ]; then
  export CUDA_VISIBLE_DEVICES=3
  python3 -m espnet2.bin.k2_asr_inference \
    --ngpu 1 \
    --data_path_and_name_and_type dump/raw/test_clean/wav.scp,speech,sound \
    --key_file dump/raw/test_clean/wav.scp \
    --asr_train_config exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/config.yaml \
    --asr_model_file exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/valid.acc.ave_10best.pth \
    --output_dir ${asr_exp}/decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean/
fi

test_sets="test_clean"
inference_config=conf/decode_asr.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
if [ $stage -le 2 ]; then
  bash asr.sh \
    --stage 12 \
    --lang en \
    --nbpe 5000 \
    --asr_exp ${asr_exp} \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --skip_data_prep false \
    --skip_train true \
    --test_sets ${test_sets} \
    --train_set "none" \
    --valid_set "none"

fi
