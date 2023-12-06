# ASR2 recipe for Libriheavy medium (with casing and punctuation)

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans1000_nBPE3000

## Environments
- date: `Fri Nov  3 12:31:34 CET 2023`
- python version: `3.10.8 (main, Nov 14 2022, 00:00:00) [GCC 11.3.1 20220421 (Red Hat 11.3.1-3)]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.0.1+cu118`
- Git hash: `b62c674b740d148a5e1d07b1bc3eb8e5dddb5839`
  - Commit date: `Mon Oct 30 13:47:12 2023 +0100`
  - ASR config: [conf/tuning/train_discrete_asr_e_branchformer1_e12_lr1e-3.yaml](conf/tuning/train_discrete_asr_e_branchformer1_e12_lr1e-3.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)
  - Pretrained model: [https://huggingface.co/espnet/akreal_lh_medium_asr2_e_branchformer_wavlm_large_21_km1k_bpe_rm6k_bpe_ts3k](espnet/akreal_lh_medium_asr2_e_branchformer_wavlm_large_21_km1k_bpe_rm6k_bpe_ts3k)

## exp/asr_train_discrete_asr_e_branchformer1_e12_lr1e-3_raw_wavlm_large_21_km1000_bpe_rm6000_bpe_ts3000
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev|5348|217854|88.9|10.2|0.9|0.5|11.6|91.1|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2557|102570|90.9|8.5|0.6|0.4|9.5|88.9|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2815|111093|87.4|11.4|1.2|0.6|13.2|91.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev|5348|1177476|97.1|1.5|1.4|0.7|3.6|91.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2557|552463|97.9|1.1|1.0|0.5|2.7|89.0|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2815|600913|96.4|1.8|1.7|0.9|4.4|91.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev|5348|349335|90.0|6.1|3.9|1.4|11.4|91.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2557|160091|92.1|4.9|3.0|1.2|9.0|89.0|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2815|181621|88.0|7.0|5.0|1.6|13.6|91.8|
