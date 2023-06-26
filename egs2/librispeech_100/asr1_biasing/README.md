# TCPGen in RNN-T
## TCPGen

## asr_train_conformer_transducer_tcpgen500_deep_sche30_suffix

- ASR Config: [conf/train_rnnt.yaml](conf/train_rnnt.yaml)
- Params: 27.13M

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|52576|95.4|4.1|0.5|0.7|5.3|50.7|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|52343|85.9|12.4|1.7|1.7|15.8|78.0|

## asr_train_conformer_transducer_tcpgen500_deep_sche30_suffix

- ASR Config: [conf/tuning/train_rnnt_std_tcpgen.yaml](conf/tuning/train_rnnt_std_tcpgen.yaml)
- Params: 26.99M

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|52576|95.1|4.5|0.5|0.7|5.6|54.3|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|52343|85.2|13.0|1.8|1.7|16.5|79.6|
