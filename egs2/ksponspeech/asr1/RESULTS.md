# RESULTS
## Dataset
- KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition
  - Database: https://aihub.or.kr/aidata/105
  - Paper: https://www.mdpi.com/2076-3417/10/19/6936
  - This corpus contains 969 h of general open-domain dialog utterances, spoken by 2000 native Korean speakers.

## Environments
- date: `Sun Aug  1 11:02:20 EDT 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `449c0f84fd0d8a851b5be29c0f7e4717a5596b98`
  - Commit date: `Fri Jul 30 11:47:21 2021 -0400`

## Conformer+Transformer LM 
### asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_clean|3000|65475|93.1|3.7|3.2|1.9|8.8|60.2|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_other|3000|92640|92.6|4.2|3.1|2.2|9.6|74.1|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_clean|3000|20401|82.0|14.2|3.8|3.6|21.6|60.2|
|decode_asr_lm_lm_train_lm_transformer3_kr_bpe2309_valid.loss.ave_asr_model_valid.acc.best/eval_other|3000|26621|78.6|17.8|3.6|4.9|26.3|74.1|
