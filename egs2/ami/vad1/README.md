## Environments
- date: `Thu May  4 10:25:40 EDT 2023`
- python version: `3.8.16 (default, Mar  2 2023, 03:21:46)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.8.1`
- Git hash: `1bd1db914b21bfb5ae5acbe2fc7162e3815ed260`
  - Commit date: `Thu May 4 08:48:15 2023 -0400`

## Model info
- Model link: https://huggingface.co/espnet/dongwei_ami_vad_rnn
- ASR config: conf/tuning/train_vad_rnn.yaml
- Decode config: conf/tuning/decode_rnn.yaml
- The metrics are frame-level precision, frame-level recall and frame-level F1 for speech frames

## exp/vad_train_asr_transformer_raw
### PRECISION

|dataset|value|
|---|---|
|exp/vad_train_asr_transformer_raw/decode_rnn_vad_model_valid.acc.ave/ihm_dev/result.txt|0.9311|
|exp/vad_train_asr_transformer_raw/decode_rnn_vad_model_valid.acc.ave/ihm_eval/result.txt|0.9547|

### RECALL

|dataset|value|
|---|---|
|exp/vad_train_asr_transformer_raw/decode_rnn_vad_model_valid.acc.ave/ihm_dev/result.txt|0.9277|
|exp/vad_train_asr_transformer_raw/decode_rnn_vad_model_valid.acc.ave/ihm_eval/result.txt|0.9412|

### F1_SCORE

|dataset|value|
|---|---|
|exp/vad_train_asr_transformer_raw/decode_rnn_vad_model_valid.acc.ave/ihm_dev/result.txt|0.9294|
|exp/vad_train_asr_transformer_raw/decode_rnn_vad_model_valid.acc.ave/ihm_eval/result.txt|0.9479|
