#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k

#==================================================
# This corpus combines five public datasets
# (VCTK+DEMAND, DNS1, WHAMR!, CHiME-4, and REVERB)
#==================================================
train_set=train_dns20_vctk_whamr_chime4_reverb
valid_set=valid_dns20_vctk_whamr_chime4
test_sets="chime4_et05_simu_isolated_6ch_track dns20_tt_synthetic_no_reverb dns20_tt_synthetic_with_reverb reverb_et_simu_8ch_multich whamr_tt_mix_single_anechoic_max_16k whamr_tt_mix_single_reverb_max_16k" # 16kHz
# test_sets="vctk_noisy_tt_2spk" # 48kHz

# Note: It is better to skip stage 3 and stage 4 if you want to
# use data of different sampling frequencies for training.
# To do so, just manually copy the data/${train_set}, data/${valid_set}, and data/${test_sets} to dump/raw/.

# Note: In the inference and scoring stages (stages 7 and 8),
# it is recommended to add the following arguments for 8 kHz data:
#   --inference_enh_config "conf/inference/test_8k.yaml" --fs 8k
# it is recommended to add the following arguments for 16 kHz data:
#   --inference_enh_config "conf/inference/test_16k.yaml" --fs 16k
# it is recommended to add the following arguments for 48 kHz data:
#   --inference_enh_config "conf/inference/test_48k.yaml" --fs 48k

# Note: If you want to set the reference channel (e.g., to 3) for inference and scoring,
# you can add the following arguments:
#   --inference_args "--normalize_output_wav true --ref_channel 3" --ref_channel 3

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 1 \
    --enh_config conf/tuning/train_enh_muses_normalizeVar_dereverb_mem_refch0.yaml \
    --use_dereverb_ref true \
    --use_noise_ref false \
    --max_wav_duration 50 \
    --inference_model "valid.loss.best.pth" \
    "$@"
