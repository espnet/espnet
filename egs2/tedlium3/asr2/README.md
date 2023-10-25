# ASR2 recipe for Tedlium3
## Related work:
   Chang, Xuankai, et al. "Exploration of Efficient End-to-End ASR using Discretized Input from Self-Supervised Learning." InterSpeech 2023.
   <details>
   <summary>bib info</summary>

   ```
   @article{chang2023exploration,
        title={Exploration of Efficient End-to-End ASR using Discretized Input from Self-Supervised Learning},
        author={Chang, Xuankai and Yan, Brian and Fujita, Yuya and Maekaku, Takashi and Watanabe, Shinji},
        journal={arXiv preprint arXiv:2305.18108},
        year={2023}
   }
   ```
   </details>


# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans1000_nBPE2000 (~14.5 hours for 35epochs with A5000 x 1)

## Environments
- date: `Thu Oct 19 22:11:12 JST 2023`
- python version: `3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `7bcdab47ff7f47e55d52061e55db4128913f32b6`
  - Commit date: `Thu Aug 31 20:42:18 2023 +0900`

## Model info
- Model link: https://huggingface.co/espnet/kohei0209_ted3_asr2_e_branchformer1_raw_wavlm_large_21_km1000_bpe_rm2000_bpe_ts500_sp
- ASR config: [conf/tuning/train_discrete_asr_e_branchformer1.yaml](conf/tuning/train_discrete_asr_e_branchformer1.yaml)
- Decode config: [conf/tuning/decode_ctc0.3.yaml](conf/tuning/decode_ctc0.3.yaml)


## exp/asr_train_discrete_asr_e_branchformer1_raw_wavlm_large_21_km1000_bpe_rm2000_bpe_ts500_sp/
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/test|1155|27500|94.6|3.4|2.0|3.5|8.9|79.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/test|1155|145066|97.4|0.9|1.7|4.2|6.7|79.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/test|1155|54206|96.1|2.2|1.7|3.8|7.7|79.0|

## exp/asr_train_discrete_asr_e_branchformer1_raw_wavlm_large_21_km1000_bpe_rm2000_bpe_ts500_sp/decode_asr_model_valid.acc.ave
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|507|17783|94.2|3.7|2.2|3.2|9.0|84.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|507|95429|97.2|0.9|1.9|3.6|6.3|84.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|507|36002|95.8|2.3|1.9|3.2|7.4|84.8|
