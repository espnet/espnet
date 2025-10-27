#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup_response
valid_set=train_dev_response
test_sets="eval2000_response"

train_config=conf/train_s2s_lr1e-5_fisher_complete.yaml
inference_config=conf/decode_asr_short2.yaml

ssl_opts="\
  --ssl_choice espnet_hubert \
  --ssl_nlayer 18 \
  --ssl_checkpoint_path /work/nvme/bbjs/arora1/speech_lm/speechlm_unified_v1_1.7B/exp/kmeans/38epoch.pth \
  --ssl_kmeans_path /work/nvme/bbjs/arora1/speech_lm/speechlm_unified_v1_1.7B/exp/kmeans/xeus_18_5000clusters/km_5000.mdl \
  --ssl_batch_bins 9800000 \
"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch"
bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B"

./speechlm.sh \
    --skip_train false \
    --task "codec_ssl_cot_full_utt2spk" \
    --data_name librispeech_100 \
    --fs 16000 \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 0.1 \
    --max_wav_duration 60.0 \
    --local_data_opts "--speechlm_data_prep true"\
    ${ssl_opts} ${codec_opts} ${bpe_opts} \
    "$@"
