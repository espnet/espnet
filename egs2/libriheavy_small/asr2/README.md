# ASR2 recipe for Libriheavy small (with casing and punctuation)

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans2000_nBPE3000

## Environments
- date: `Fri Nov  3 12:26:08 CET 2023`
- python version: `3.10.8 (main, Nov 14 2022, 00:00:00) [GCC 11.3.1 20220421 (Red Hat 11.3.1-3)]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.0.1+cu118`
- Git hash: `b62c674b740d148a5e1d07b1bc3eb8e5dddb5839`
  - Commit date: `Mon Oct 30 13:47:12 2023 +0100`
  - ASR config: [conf/tuning/train_discrete_asr_e_branchformer1_e12_lr1e-3.yaml](conf/tuning/train_discrete_asr_e_branchformer1_e12_lr1e-3.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)
  - Pretrained model: [https://huggingface.co/espnet/akreal_lh_small_asr2_e_branchformer_wavlm_large_21_km2k_bpe_rm6k_bpe_ts3k_sp](espnet/akreal_lh_small_asr2_e_branchformer_wavlm_large_21_km2k_bpe_rm6k_bpe_ts3k_sp)

## exp/asr_train_discrete_asr_e_branchformer1_e12_lr1e-3_raw_wavlm_large_21_km2000_bpe_rm6000_bpe_ts3000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev|5348|217854|87.2|11.8|1.0|0.6|13.4|93.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2557|102570|89.5|9.9|0.6|0.5|11.0|91.3|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2815|111093|85.3|13.4|1.3|0.7|15.4|95.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev|5348|1177476|96.7|1.8|1.5|0.8|4.1|93.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2557|552463|97.5|1.4|1.1|0.6|3.1|91.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2815|600913|95.9|2.2|1.9|1.1|5.2|95.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev|5348|353415|88.2|7.2|4.6|1.8|13.5|93.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2557|161841|90.6|5.9|3.5|1.4|10.8|91.4|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2815|183779|85.9|8.4|5.6|2.1|16.1|95.0|
