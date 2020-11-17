# The first conformer result
## Environments
- date: `Mon Nov 16 18:59:34 EST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.4.0`
- Git hash: `eda997f9e97ad959c6b13df1b34eb24fb8c52768`
  - Commit date: `Thu Oct 8 07:32:49 2020 -0400`

## asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp
- https://zenodo.org/record/4276519
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.0|1.8|0.2|0.3|2.3|27.6|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.0|4.3|0.6|0.5|5.5|45.1|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|28.2|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|94.9|4.5|0.7|0.6|5.8|48.6|

# The second tentative result
## Environments
- date: `Mon Jul 20 04:38:33 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `7b37518e0c7bc0c550a80b83fff4b3e927ebb142`
  - Commit date: `Wed Jul 8 17:25:03 2020 -0400`

- Update from the previous result
  - ASR: dropout 0.1, encoder 18 layers, subsampling (2x3=6), and speed perturbation
  - LM: Adam optimizer

## asr_train_asr_transformer_e18_raw_bpe_sp
- https://zenodo.org/record/3966501
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2703|54402|97.9|1.9|0.2|0.2|2.3|28.5|
|decode_dev_other_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2864|50948|94.7|4.6|0.6|0.6|5.9|46.0|
|decode_test_clean_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2620|52576|97.7|2.0|0.3|0.3|2.5|30.0|
|decode_test_other_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2939|52343|94.5|4.8|0.7|0.7|6.2|50.0|


# The first tentative result
## Environments
- date: `Fri Jul  3 04:48:06 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `af9cb2449a15b89490964b413a9a02422a26fa5e`
  - Commit date: `Thu Jul 2 07:56:03 2020 -0400`

- beam size 20 (60 in the best system), ASR epoch ~60, LM epoch 2

## asr_train_asr_transformer_raw_bpe_optim_conflr0.001
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2703|54402|97.2|2.4|0.3|0.3|3.1|35.2|
|decode_dev_other_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2864|50948|93.2|6.0|0.8|0.8|7.6|54.7|
|decode_test_clean_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2620|52576|97.0|2.6|0.4|0.4|3.4|37.4|
|decode_test_other_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2939|52343|92.9|6.1|1.0|0.9|8.0|58.3|
