# E-Branchformer + Transformer (joint CTC/attention), bpe1000 + RNN LM
## Environments
- date: `Thu Jun 18 05:59:41 EDT 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `295ef69aed650d97ce44088ce2ee1675a6b287db`
  - Commit date: `Mon Jun 1 07:25:42 2026 -0400`

## Results
- Model link: (to be uploaded to HuggingFace)
- ASR config: [conf/tuning/train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml](conf/tuning/train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml)
- LM config: [conf/train_lm.yaml](conf/train_lm.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_kn_bpe1000_valid.loss.ave_asr_model_valid.acc.ave/test|21891|319118|80.4|18.2|1.4|4.8|24.4|83.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_kn_bpe1000_valid.loss.ave_asr_model_valid.acc.ave/test|21891|2634207|96.8|1.2|1.9|0.7|3.8|83.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_kn_bpe1000_valid.loss.ave_asr_model_valid.acc.ave/test|21891|927346|91.5|4.9|3.6|1.3|9.8|70.7|
