# How2 corpus
## HOW TO GET AND USE HOW2:

1. Go to https://github.com/srvk/how2-dataset and follow instructions.
For ASR, the following parts should be selected when filling form:
```
   (audio) fbank+pitch features in Kaldi scp/ark format
   (en) English text
```

First part (feats), contains directory 'fbank_pitch_181516'.
Second part (text) contains 'how2-300h-v1' with following directories:

```
   how2-300h-v1
   |_ data/
      |_ val
      |_ train
      |_ dev5
   |_ features/
```

2. Set up 'HOW2_FEATS' and 'HOW_TEXT' in db.sh to where you put 'how-300h-v1'
   and 'fbank_pitch_181516' directories.


# RESULTS:

## asr_train_rnn_extracted_bpe
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev5decode_asr_model_valid.acc.best|2305|41013|87.1|9.7|3.2|3.1|15.9|73.9|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|45040|87.0|9.7|3.2|3.2|16.1|75.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev5decode_asr_model_valid.acc.best|2305|203929|94.0|2.6|3.5|3.0|9.0|73.9|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|224550|93.8|2.7|3.5|3.1|9.3|75.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev5decode_asr_model_valid.acc.best|2305|60721|88.0|7.6|4.4|3.2|15.3|73.9|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|66657|87.2|7.9|4.9|3.2|16.0|75.7|
