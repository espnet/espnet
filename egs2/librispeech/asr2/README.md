# ASR2 recipe for LibriSpeech960
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

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans1000_nBPE5000 (18.5 hours for 35epochs with A100 x 1)

## Environments
- date: `Fri Jun 23 09:33:15 CDT 2023`
- python version: `3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.13.1`
161e4bb5092b77361fcb22e28691d028ef7a7194 (HEAD -> master, espnet/master)
Merge: b5a88e954 5daff333a
Author: mergify[bot] <37929162+mergify[bot]@users.noreply.github.com>
Date:   Thu Jun 22 14:16:13 2023 +0000
- Git hash: `161e4bb5092b77361fcb22e28691d028ef7a7194`
  - Commit date: `Thu Jun 22 14:16:13 2023 +0000`
  - ASR config: [conf/tuning/train_discrete_asr_e_branchformer1_1gpu.yaml](conf/tuning/train_discrete_asr_e_branchformer1_1gpu.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)
  - Pretrained model: [https://huggingface.co/espnet/simpleoier_ls960_asr2_train_e_branchformer1_1gpu_raw_wavlm_large_21_km1k_bpe_rm5k_bpe_ts5k_sp](https://huggingface.co/espnet/simpleoier_ls960_asr2_train_e_branchformer1_1gpu_raw_wavlm_large_21_km1k_bpe_rm5k_bpe_ts5k_sp)

## exp/asr_train_discrete_asr_e_branchformer1_1gpu_raw_wavlm_large_21_km1000_bpe_rm5000_bpe_ts5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|98.1|1.8|0.2|0.2|2.1|27.6|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|95.8|3.9|0.3|0.4|4.6|42.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|29.3|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|95.9|3.8|0.3|0.5|4.6|43.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|27.6|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|265951|98.5|0.9|0.6|0.5|1.9|42.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.3|0.2|0.2|0.7|29.3|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|272758|98.7|0.7|0.5|0.5|1.8|43.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|68010|97.6|1.7|0.7|0.3|2.7|27.6|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|63110|94.5|4.1|1.4|0.7|6.2|42.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|65818|97.2|1.8|0.9|0.3|3.1|29.3|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|65101|94.7|3.6|1.7|0.7|6.0|43.6|

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans1000_nBPE5000 using Conv_Subsample of 3 (13.8 hours for 35epochs with A100 x 1)

## Environments
- date: `Fri Jun 23 10:16:00 CDT 2023`
- python version: `3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.13.1`
- Git hash: `ac5ffad321c64cbf159e4b789dbe26c43dc31b0e`
  - Commit date: `Wed Jun 14 19:50:52 2023 +0000`
  - ASR config: [conf/tuning/train_discrete_asr_e_branchformer1_conv1d3_1gpu.yaml](conf/tuning/train_discrete_asr_e_branchformer1_conv1d3_1gpu.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)
  - Pretrained model: [https://huggingface.co/espnet/simpleoier_ls960_asr2_e_branchformer1_conv1d3_1gpu_raw_wavlm_large_21_km1k_bpe_rm5k_bpe_ts5k_sp](https://huggingface.co/espnet/simpleoier_ls960_asr2_e_branchformer1_conv1d3_1gpu_raw_wavlm_large_21_km1k_bpe_rm5k_bpe_ts5k_sp)

## exp/asr_train_discrete_asr_e_branchformer1_conv1d3_1gpu_raw_wavlm_large_21_km1000_bpe_rm5000_bpe_ts5000_sp
### WER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|97.8|1.9|0.2|0.3|2.4|30.0|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|95.6|4.0|0.4|0.4|4.8|43.8|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|97.8|1.9|0.3|0.3|2.5|30.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|95.6|4.0|0.3|0.5|4.8|46.3|
### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|288456|99.4|0.3|0.3|0.2|0.8|30.0|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|265951|98.4|0.9|0.7|0.5|2.1|43.8|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|281530|99.4|0.3|0.3|0.2|0.8|30.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|272758|98.6|0.8|0.6|0.5|1.9|46.3|
### TER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|68010|97.2|1.9|0.9|0.3|3.1|30.0|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|63110|94.3|4.2|1.5|0.8|6.5|43.8|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|65818|97.1|1.9|1.0|0.3|3.2|30.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|65101|94.4|3.8|1.7|0.7|6.3|46.3|

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans2000_nBPE6000 (12 hours for 35 epochs with A100 x 2)

## Environments
- date: `Thu Jun 22 17:45:33 EDT 2023`
- python version: `3.9.16 (main, Mar  8 2023, 14:00:05)  [GCC 11.2.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.13.1`
- Git hash: `ac5ffad321c64cbf159e4b789dbe26c43dc31b0e`
  - Commit date: `Wed Jun 14 19:50:52 2023 +0000`
  - ASR config: [conf/tuning/train_discrete_asr_e_branchformer1.yaml](conf/tuning/train_discrete_asr_e_branchformer1.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)
  - Pretrained model: [https://huggingface.co/espnet/simpleoier_ls960_asr2_train_e_branchformer1_raw_wavlm_large_21_km2000_bpe_rm6000_bpe_ts5000_sp](https://huggingface.co/espnet/simpleoier_ls960_asr2_train_e_branchformer1_raw_wavlm_large_21_km2000_bpe_rm6000_bpe_ts5000_sp)

## exp/asr_train_discrete_asr_e_branchformer1_raw_wavlm_large_21_km2000_bpe_rm6000_bpe_ts5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|54402|98.0|1.8|0.2|0.2|2.2|27.5|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|50948|95.9|3.7|0.3|0.4|4.5|41.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|29.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|52343|95.8|3.8|0.4|0.5|4.6|43.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|288456|99.4|0.3|0.3|0.2|0.8|27.5|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|265951|98.5|0.9|0.6|0.4|1.9|41.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.3|0.3|0.2|0.7|29.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|272758|98.7|0.7|0.6|0.5|1.8|43.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_clean|2703|68049|97.4|1.8|0.8|0.3|2.9|27.5|
|decode_ctc0.3_asr_model_valid.acc.ave/dev_other|2864|63048|94.6|3.9|1.5|0.6|6.0|41.7|
|decode_ctc0.3_asr_model_valid.acc.ave/test_clean|2620|65769|97.3|1.8|1.0|0.3|3.0|29.2|
|decode_ctc0.3_asr_model_valid.acc.ave/test_other|2939|65268|94.6|3.6|1.8|0.6|5.9|43.3|
