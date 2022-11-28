# MultiIRIS (WPD-beamformer-Enh, WavLM_Large + SpecAug + Conformer-ASR with Transformer-LM)

## Notes
- Joint finetuning requires pre-trained Enh and ASR models.
- `local/run_multiiris.sh` performs (1) pre-training of the Enh model, (2) pre-training of the ASR model, and (3) joint finetuning of the entire system.
- The language model is also trained on (2).
- The final scores will be summarized in `enh_asr1/exp/*` similar to `run.sh`.

## Environments
- date: `Tue Oct 11 02:40:53 UTC 2022`
- python version: `3.7.4 (default, Aug 13 2019, 20:35:49)  [GCC 7.3.0]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.10.1+cu111`
- Git hash: `8ed83f45d5aa2ca6b3635e44b9c29afb9b5fb600`
  - Commit date: `Tue Oct 11 18:59:57 2022 +0900`

## enh_asr_train_enh_asr_wpd_init_noenhloss_wavlm_conformer_raw_en_char
- Enh pre-training config: [../enh1/conf/tuning/train_enh_beamformer_wpd_ci_sdr_shorttap.yaml](../enh1/conf/tuning/train_enh_beamformer_wpd_ci_sdr_shorttap.yaml)
- ASR pre-training config: [../asr1/conf/tuning/train_asr_conformer_wavlm2.yaml](../asr1/conf/tuning/train_asr_conformer_wavlm2.yaml)
- Enh-ASR finetuning config: [./conf/tuning/train_enh_asr_wpd_init_noenhloss_wavlm_conformer.yaml](./conf/tuning/train_enh_asr_wpd_init_noenhloss_wavlm_conformer.yaml)
- LM config: [../asr1/conf/train_lm_transformer.yaml](../asr1/conf/train_lm_transformer.yaml)
- Pretrained model: [https://huggingface.co/Yoshiki/chime4_enh_asr1_wpd_wavlm_conformer](https://huggingface.co/Yoshiki/chime4_enh_asr1_wpd_wavlm_conformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/dt05_real_isolated_6ch_track|1640|27119|98.8|0.9|0.2|0.2|1.3|16.2|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/dt05_simu_isolated_6ch_track|1640|27120|98.9|0.9|0.2|0.1|1.3|15.2|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/et05_real_isolated_6ch_track|1320|21409|98.4|1.4|0.2|0.2|1.8|20.6|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/et05_simu_isolated_6ch_track|1320|21416|98.9|1.0|0.2|0.1|1.2|15.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/dt05_real_isolated_6ch_track|1640|160390|99.7|0.1|0.2|0.2|0.5|16.2|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/dt05_simu_isolated_6ch_track|1640|160400|99.7|0.1|0.2|0.1|0.5|15.2|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/et05_real_isolated_6ch_track|1320|126796|99.5|0.2|0.3|0.2|0.7|20.6|
|decode_asr_transformer_largelm_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave_10best/et05_simu_isolated_6ch_track|1320|126812|99.7|0.2|0.2|0.1|0.5|15.2|

### Enhancement

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_dt05_simu_isolated_6ch_track|94.48|14.95|14.95|0.00|12.43|
|enhanced_et05_simu_isolated_6ch_track|94.93|16.08|16.08|0.00|13.98|


# RESULTS
## Environments
- date: `Thu Apr 28 00:09:17 EDT 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.8.1`
- Git hash: `44971ff962aae30c962226f1ba3d87de057ac00e`
  - Commit date: `Wed Apr 27 10:13:03 2022 -0400`

## enh_asr_train_enh_asr_convtasnet_init_noenhloss_wavlm_transformer_init_lr1e-4_accum1_adam_specaug_bypass0_raw_en_char
- Pretrained model: https://huggingface.co/espnet/simpleoier_chime4_enh_asr_convtasnet_init_noenhloss_wavlm_transformer_init_raw_en_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|27119|98.3|1.3|0.4|0.2|1.9|21.8|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_2mics|1640|27119|98.5|1.2|0.3|0.2|1.7|19.6|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|27119|98.6|1.1|0.3|0.2|1.5|18.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|27120|97.2|2.1|0.7|0.3|3.1|28.9|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_2mics|1640|27120|97.9|1.5|0.5|0.2|2.3|25.2|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|27120|98.4|1.2|0.4|0.1|1.7|19.9|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|21409|96.7|2.6|0.7|0.4|3.7|31.6|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_2mics|1320|21409|97.4|2.0|0.6|0.3|2.9|27.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|21409|97.8|1.8|0.4|0.2|2.5|24.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|21416|94.6|3.7|1.6|0.5|5.9|37.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_2mics|1320|21416|96.6|2.5|1.0|0.3|3.7|32.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|21416|97.5|1.9|0.7|0.3|2.9|28.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|160390|99.4|0.2|0.4|0.2|0.8|21.8|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_2mics|1640|160390|99.5|0.2|0.3|0.2|0.7|19.6|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|160390|99.6|0.1|0.3|0.2|0.6|18.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|160400|98.8|0.5|0.7|0.3|1.5|28.9|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_2mics|1640|160400|99.2|0.3|0.5|0.2|1.1|25.2|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|160400|99.5|0.2|0.3|0.1|0.7|19.9|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|126796|98.6|0.6|0.8|0.4|1.8|31.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_2mics|1320|126796|98.9|0.4|0.7|0.3|1.4|27.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|126796|99.1|0.4|0.5|0.2|1.1|24.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|126812|97.0|1.2|1.9|0.6|3.7|37.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_2mics|1320|126812|98.2|0.6|1.1|0.4|2.1|32.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|126812|98.8|0.4|0.8|0.3|1.5|28.9|

### Enhancement

|dataset|STOI|SDR|SI_SNR|
|---|---|---|---|
|dt05_simu_isolated_1ch_track|0.86|4.97|1.77|
|et05_simu_isolated_1ch_track|0.85|5.45|0.88|


## enh_asr_train_enh_asr_convtasnet_fbank_transformer_raw_en_char
- Pretrained model: https://huggingface.co/espnet/simpleoier_chime4_enh_asr_train_enh_asr_convtasnet_fbank_transformer_raw_en_char

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|27119|91.8|6.0|2.2|0.8|9.0|57.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_2mics|1640|27119|93.0|5.2|1.8|0.6|7.7|53.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|27119|93.9|4.5|1.6|0.5|6.7|49.9|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|27120|89.9|7.6|2.4|1.0|11.1|59.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_2mics|1640|27120|92.2|6.0|1.9|0.7|8.6|55.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|27120|93.6|4.9|1.5|0.6|7.1|51.6|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|21409|84.6|11.4|4.0|1.5|17.0|69.4|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_2mics|1320|21409|86.7|9.7|3.5|1.3|14.5|64.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|21409|89.2|7.9|2.9|1.0|11.8|61.2|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|21416|82.8|13.1|4.1|1.9|19.1|69.4|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_2mics|1320|21416|86.0|10.5|3.5|1.5|15.5|67.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|21416|88.1|8.9|3.1|1.2|13.1|64.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|160390|95.9|1.7|2.3|0.8|4.8|57.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_2mics|1640|160390|96.6|1.4|2.0|0.6|4.0|53.3|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|160390|97.1|1.1|1.8|0.5|3.4|49.9|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|160400|94.7|2.5|2.9|1.0|6.3|59.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_2mics|1640|160400|95.9|1.7|2.3|0.7|4.8|55.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|160400|96.8|1.4|1.9|0.6|3.8|51.6|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|126796|91.5|3.8|4.6|1.6|10.0|69.4|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_2mics|1320|126796|92.8|3.2|4.0|1.2|8.4|64.7|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|126796|94.3|2.4|3.3|1.0|6.6|61.2|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|126812|90.3|4.8|4.9|2.2|11.9|69.4|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_2mics|1320|126812|92.2|3.5|4.2|1.7|9.5|67.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|126812|93.7|2.7|3.5|1.4|7.7|64.8|

### Enhancement

|dataset|STOI|SDR|SI_SNR|
|---|---|---|---|
|dt05_simu_isolated_1ch_track|0.87|7.14|4.51|
|et05_simu_isolated_1ch_track|0.85|7.47|3.02|

