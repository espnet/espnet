# ASR for phonetic transcription (IPA + punctuation)

Purpose of this system is to produce output that is suitable as a direct input for a phone based TTS
([paper](https://arxiv.org/abs/2207.04834)).

## Environments
- date: `Sat Oct 14 03:02:09 CEST 2023`
- python version: `3.10.8 (main, Nov 14 2022, 00:00:00) [GCC 11.3.1 20220421 (Red Hat 11.3.1-3)]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 2.0.1+cu118`
- Git hash: `3800a13ae8972839d506b85585c41e6b24daf812`
  - Commit date: `Sun Oct 8 17:51:17 2023 +0200`

## exp/asr_train_asr_raw_en_bpe100_sp

- ASR Config: [conf/tuning/train_asr_e_branchformer.yaml](conf/tuning/train_asr_e_branchformer.yaml)
- Params: 141.39M
- Model link: [https://huggingface.co/espnet/akreal_libritts_asr_phn](https://huggingface.co/espnet/akreal_libritts_asr_phn)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev-clean|5736|95872|91.7|8.0|0.4|0.8|9.1|67.0|
|decode_asr_asr_model_valid.acc.ave/dev-other|4613|69577|88.5|10.9|0.6|1.2|12.7|74.2|
|decode_asr_asr_model_valid.acc.ave/test-clean|4837|87078|91.4|8.2|0.4|0.8|9.4|70.4|
|decode_asr_asr_model_valid.acc.ave/test-other|5120|72541|87.0|12.2|0.8|1.1|14.1|77.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev-clean|5736|570710|98.4|0.8|0.9|0.6|2.2|67.1|
|decode_asr_asr_model_valid.acc.ave/dev-other|4613|414781|97.2|1.6|1.2|1.0|3.8|74.2|
|decode_asr_asr_model_valid.acc.ave/test-clean|4837|530647|98.5|0.7|0.8|0.6|2.2|70.5|
|decode_asr_asr_model_valid.acc.ave/test-other|5120|429463|96.7|1.7|1.6|1.0|4.3|77.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev-clean|5736|433548|97.6|1.4|1.0|0.6|3.0|67.1|
|decode_asr_asr_model_valid.acc.ave/dev-other|4613|316550|96.1|2.5|1.4|1.0|5.0|74.2|
|decode_asr_asr_model_valid.acc.ave/test-clean|4837|404031|97.7|1.4|0.9|0.7|2.9|70.5|
|decode_asr_asr_model_valid.acc.ave/test-other|5120|327248|95.4|2.8|1.8|1.1|5.7|77.1|

