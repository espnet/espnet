# ASR2 recipe for Gigaspeech
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

# E-Branchformer ASR2 Discrete tokens with WavLM_large_Layer21_Kmeans1000_nBPE6000 


## exp/asr_train_discrete_asr_e_branchformer1_1gpu_raw_wavlm_large_21_km1000_bpe_rm6000_bpe_ts5000 (5.5hrs/epoch with V100:16 x 2) (10th epoch result)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.best/dev|5715|129240|88.3|6.7|5.0|2.2|13.8|82.0|
|decode_ctc0.3_asr_model_valid.acc.best/test|19930|392325|88.9|7.2|4.8|1.8|13.8|76.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.best/dev|5715|673778|94.7|1.3|3.9|1.7|7.0|82.0|
|decode_ctc0.3_asr_model_valid.acc.best/test|19930|2056231|94.4|1.6|4.0|1.4|7.0|76.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.best/dev|5715|160740|87.3|5.6|7.2|1.8|14.5|82.0|
|decode_ctc0.3_asr_model_valid.acc.best/test|19930|493006|86.8|6.1|7.1|1.5|14.8|76.0|

## asr_train_discrete_asr_e_branchformer1_e17_size512_mlp3072_linear1024_layerdrop_raw_wavlm_large_21_km1000_bpe_rm6000_bpe_ts5000 (14hrs/epoch with V100:16 x 4) (12th epoch result)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.best/dev|5715|129240|88.0|7.0|5.0|2.4|14.4|82.4|
|decode_ctc0.3_asr_model_valid.acc.best/test|19930|392325|87.4||5.0|1.9|14.5|77.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.best/dev|5715|673778|94.6|1.4|4.1|1.9|7.3|82.4|
|decode_ctc0.3_asr_model_valid.acc.best/test|19930|2056231|94.1|1.6|4.3|1.5|7.5|77.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.best/dev|5715|160740|86.8|5.7|7.5|1.9|15.1|82.4|
|decode_ctc0.3_asr_model_valid.acc.best/test|19930|493006|85.9|6.3|7.8|1.6|15.7|77.2|

## ASR1 baseline
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|5715|127790|92.2|5.7|2.0|2.8|10.6|69.9|
|decode_asr_asr_model_valid.acc.ave/test|19930|390744|91.5|6.4|2.1|2.0|10.5|63.3|