# RESULTS
## Dataset
- KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition
  - Database: https://aihub.or.kr/aidata/105
  - Paper: https://www.mdpi.com/2076-3417/10/19/6936
  - This corpus contains 969 h of general open-domain dialog utterances, spoken by 2000 native Korean speakers.

## Environments
- date: `Mon Aug  2 17:20:52 EDT 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `62a7dd6d5f08c4c7b1c72a8785820fc70c9ad603`
  - Commit date: `Mon Aug 2 14:15:45 2021 -0400`
- Pretrained Model: https://zenodo.org/record/5154341

## Conformer+Transformer LM
### asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309
- Total number of ASR model parameters: 112.01 M
- Total number of LM model parameters: 51.99 M
- ASR config: `conf/tuning/train_asr_conformer8_n_fft512_hop_length256.yaml`
- LM config: `conf/tuning/train_lm_transformer3.yaml`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_clean|3000|65475|93.5|3.8|2.7|2.2|8.7|60.4|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_other|3000|92640|93.1|4.3|2.6|2.5|9.4|74.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_clean|3000|20401|82.3|14.5|3.2|4.0|21.7|60.4|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_other|3000|26621|79.0|18.0|3.0|5.6|26.6|74.0|
