# ASR2 recipe for CoVoST2 (es-en)
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

# E-Branchformer ASR2 Discrete tokens with mHuBERT_Layer9_Kmeans1000_nBPE5000

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
  - ASR config: [conf/tuning/train_discrete_asr_e_branchformer1_2gpu_km1000.yaml](conf/tuning/train_discrete_asr_e_branchformer1_2gpu_km1000.yaml)
  - Decode config: [conf/decode_ctc0.3.yaml](conf/decode_ctc0.3.yaml)

## exp/asr_train_discrete_asr_e_branchformer1_2gpu_km1000_raw_es_mhubert_base_vp_en_es_fr_it3_9_km1000_bpe_rm5000_bpe_ts1000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/test.es-en|13221|130157|82.1|15.3|2.6|1.8|19.7|73.0|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev.es-en|13221|129002|83.4|14.2|2.4|1.5|18.2|71.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/test.es-en|13221|779525|93.5|3.3|3.2|1.5|7.9|73.0|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev.es-en|13221|769107|94.2|2.8|3.0|1.2|7.1|71.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc0.3_asr_model_valid.acc.ave/test.es-en|13221|268326|83.7|10.7|5.7|1.5|17.9|73.0|
|decode_ctc0.3_asr_model_valid.acc.ave/org/dev.es-en|13221|265001|85.0|9.7|5.3|1.3|16.3|71.4|
