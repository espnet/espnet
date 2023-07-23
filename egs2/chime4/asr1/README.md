# E-Branchformer: 10 layers
## Environments
- date: `Wed Dec 28 15:49:24 EST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `f9a8009aef6ff9ba192a78c19b619ae4a9f3b9d2`
  - Commit date: `Wed Dec 28 00:30:54 2022 -0500`

## asr_train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k_raw_en_char_sp
- ASR config: [conf/tuning/train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k.yaml](conf/tuning/train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k.yaml)
- Params: 30.79M
- LM config: [conf/train_lm_transformer.yaml](conf/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/chime4_e_branchformer_e10](https://huggingface.co/pyf98/chime4_e_branchformer_e10)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|27119|93.7|5.0|1.2|0.6|6.8|52.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|27120|92.4|6.1|1.6|0.7|8.4|58.2|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|21409|90.2|8.0|1.8|1.0|10.8|60.2|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|21416|88.4|9.3|2.4|1.4|13.0|66.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|160390|97.4|1.3|1.3|0.7|3.3|52.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|160400|96.6|1.8|1.7|0.9|4.3|58.2|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|126796|95.7|2.3|2.0|1.1|5.4|60.2|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|126812|94.4|2.8|2.8|1.5|7.2|66.1|


## Whisper [medium_finetuning](conf/tuning/train_asr_whisper_full_warmup1500.yaml) without LM

## Environments
- date: `Fri Jul 21 12:47:17 JST 2023`
- python version: `3.10.10 (main, Mar 21 2023, 18:45:11) [GCC 11.2.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.13.1`
- Git hash: `d7172fcb7181ffdcca9c0061400254b63e37bf21`
  - Commit date: `Sat Jul 15 15:01:30 2023 +0900`
- Pretrained URL: [espnet/yoshiki_chime4_whisper_medium_finetuning](https://huggingface.co/espnet/yoshiki_chime4_whisper_medium_finetuning)

- token_type: whisper_multilingual
- cleaner: whisper_en

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|24791|97.7|1.9|0.5|0.7|3.0|25.7|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|24792|95.9|3.3|0.8|0.8|4.9|37.0|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|19341|96.3|3.2|0.5|0.8|4.5|33.6|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|19344|93.1|5.8|1.1|1.2|8.1|43.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|141889|99.2|0.4|0.4|0.7|1.5|25.7|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|141900|98.2|0.9|0.9|0.8|2.6|37.0|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|110558|98.6|0.8|0.6|0.7|2.1|33.6|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|110572|96.5|1.9|1.5|1.2|4.7|43.3|


# Conformer: 12 layers, 2048 linear units
## Environments
- date: `Wed Dec 28 20:41:40 EST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `ad91279f0108d54bd22abe29671b376f048822c5`
  - Commit date: `Wed Dec 28 20:15:42 2022 -0500`

## asr_train_asr_conformer_e12_linear2048_raw_en_char_sp
- ASR config: [conf/tuning/train_asr_conformer_e12_linear2048.yaml](conf/tuning/train_asr_conformer_e12_linear2048.yaml)
- Params: 43.04M
- LM config: [conf/train_lm_transformer.yaml](conf/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/chime4_conformer_e12_linear2048](https://huggingface.co/pyf98/chime4_conformer_e12_linear2048)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|27119|93.3|5.4|1.3|0.5|7.3|55.6|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|27120|91.7|6.7|1.6|0.9|9.1|62.0|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|21409|89.2|8.9|1.9|1.1|12.0|64.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|21416|87.8|9.6|2.6|1.4|13.6|68.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|160390|97.2|1.5|1.3|0.7|3.5|55.6|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|160400|96.3|2.0|1.7|1.0|4.7|62.0|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|126796|95.1|2.8|2.1|1.2|6.1|64.6|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|126812|94.0|3.1|3.0|1.6|7.7|68.1|



# Conformer: 12 layers, 1024 linear units
## Environments
- date: `Wed Dec 28 15:49:24 EST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `f9a8009aef6ff9ba192a78c19b619ae4a9f3b9d2`
  - Commit date: `Wed Dec 28 00:30:54 2022 -0500`

## asr_train_asr_conformer_e12_linear1024_raw_en_char_sp
- ASR config: [conf/tuning/train_asr_conformer_e12_linear1024.yaml](conf/tuning/train_asr_conformer_e12_linear1024.yaml)
- Params: 30.43M
- LM config: [conf/train_lm_transformer.yaml](conf/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/chime4_conformer_e12_linear1024](https://huggingface.co/pyf98/chime4_conformer_e12_linear1024)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|27119|92.8|5.8|1.5|0.6|7.8|56.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|27120|91.3|6.7|2.0|0.8|9.5|60.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|21409|88.6|9.2|2.1|1.2|12.5|63.8|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|21416|86.5|10.4|3.1|1.3|14.8|70.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_real_beamformit_5mics|1640|160390|96.9|1.6|1.5|0.7|3.8|56.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt05_simu_beamformit_5mics|1640|160400|96.0|2.0|2.0|1.0|4.9|60.5|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_real_beamformit_5mics|1320|126796|94.8|2.8|2.3|1.2|6.4|63.9|
|decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et05_simu_beamformit_5mics|1320|126812|93.1|3.4|3.4|1.5|8.4|70.9|



# RNN, fbank_pitch
## Environments
- date: `Sun Mar  1 17:52:59 EST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.7.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `83eb81ca220f8189f94a173851934acf8bbba0df`
  - Commit date: `Tue Jan 28 09:34:59 2020 -0500`

## asr_train_rnn_fbank_pitch_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt05_multi_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|3280|54239|73.1|22.7|4.2|3.6|30.5|92.2|
|decode_dt05_real_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1635|27011|77.4|19.1|3.4|3.2|25.7|90.5|
|decode_dt05_real_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|27119|80.8|16.2|3.0|2.5|21.7|88.0|
|decode_dt05_real_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|27119|73.5|22.3|4.1|3.3|29.8|92.5|
|decode_dt05_simu_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1635|27032|76.0|20.2|3.8|3.0|27.0|89.7|
|decode_dt05_simu_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|27120|79.4|17.3|3.4|2.3|22.9|86.7|
|decode_dt05_simu_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|27120|72.6|23.0|4.3|3.7|31.1|91.6|
|decode_et05_real_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1316|21330|64.5|29.7|5.8|4.4|39.9|94.1|
|decode_et05_real_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|21409|69.7|25.2|5.1|3.7|34.0|92.3|
|decode_et05_real_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|21409|60.2|33.2|6.6|4.9|44.7|96.1|
|decode_et05_simu_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1316|21342|65.3|29.1|5.7|4.4|39.1|94.3|
|decode_et05_simu_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|21416|68.7|26.3|4.9|4.2|35.5|93.8|
|decode_et05_simu_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|21416|63.1|30.8|6.1|4.7|41.7|94.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt05_multi_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|3280|320790|86.6|7.6|5.8|3.3|16.7|92.2|
|decode_dt05_real_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1635|159776|89.4|5.8|4.8|2.6|13.2|90.5|
|decode_dt05_real_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|160390|91.2|4.7|4.1|2.1|10.8|88.0|
|decode_dt05_real_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|160390|87.1|7.3|5.7|3.0|16.0|92.5|
|decode_dt05_simu_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1635|159876|88.3|6.5|5.3|2.8|14.6|89.7|
|decode_dt05_simu_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|160400|90.3|5.2|4.5|2.2|11.9|86.7|
|decode_dt05_simu_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1640|160400|86.2|7.9|5.9|3.5|17.3|91.6|
|decode_et05_real_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1316|126318|81.4|10.2|8.4|4.3|22.9|94.1|
|decode_et05_real_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|126796|84.5|8.3|7.1|3.4|18.9|92.3|
|decode_et05_real_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|126796|78.5|12.0|9.5|5.0|26.5|96.1|
|decode_et05_simu_beamformit_2micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1316|126381|81.4|10.2|8.4|4.3|22.9|94.3|
|decode_et05_simu_beamformit_5micsdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|126812|83.8|8.8|7.4|3.8|20.0|93.8|
|decode_et05_simu_isolated_1ch_trackdecode_rnn_lm_valid.loss.best_asr_model_valid.loss.best|1320|126812|79.9|11.0|9.0|4.9|25.0|94.5|



<!-- Generated by scripts/utils/show_asr_result.sh -->
# RESULTS
## Environments
- date: `Tue Jan 10 04:15:30 CST 2023`
- python version: `3.9.13 (main, Aug 25 2022, 23:26:10)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `d89be931dcc8f61437ac49cbe39a773f2054c50c`
  - Commit date: `Mon Jan 9 11:06:45 2023 -0600`

## asr_whisper_medium_lr1e-5_adamw_wd1e-2_3epochs

- Huggingface model URL: https://huggingface.co/espnet/shihlun_asr_whisper_medium_finetuned_chime4

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|24791|97.8|1.7|0.5|0.3|2.5|24.5|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|24792|96.1|3.0|0.9|0.5|4.4|35.6|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|19341|96.4|2.9|0.7|0.5|4.1|33.0|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|19344|93.4|5.0|1.7|0.8|7.4|41.8|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|24791|97.7|1.8|0.5|0.4|2.8|25.5|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|24792|96.0|3.3|0.8|0.7|4.8|36.0|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|19341|96.1|3.3|0.6|0.7|4.6|34.9|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|19344|92.9|5.8|1.3|1.2|8.3|43.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|141889|99.1|0.3|0.5|0.3|1.2|24.5|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|141900|98.2|0.8|1.0|0.5|2.3|35.6|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|110558|98.5|0.7|0.8|0.5|1.9|33.0|
|decode_asr_whisper_noctc_beam20_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|110572|96.5|1.6|1.9|0.8|4.3|41.8|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_real_isolated_1ch_track|1640|141889|99.1|0.4|0.5|0.5|1.3|25.5|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dt05_simu_isolated_1ch_track|1640|141900|98.2|0.9|0.9|0.6|2.4|36.0|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_real_isolated_1ch_track|1320|110558|98.4|0.9|0.7|0.6|2.2|34.9|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/et05_simu_isolated_1ch_track|1320|110572|96.3|2.0|1.7|1.2|4.9|43.2|
