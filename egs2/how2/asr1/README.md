```
HOW TO GET AND USE HOW2:

1. Go to https://github.com/srvk/how2-dataset and follow instructions.
For ASR, the following parts should be selected in form:
   (audio) fbank+pitch features in Kaldi scp/ark format
   (en) English text

First part (feats), contains directory 'fbank_pitch_181516'.
Second part (text) contains 'how2-300h-v1' with following directories:
   how2-300h-v1
   |_ data/
      |_ val
      |_ train
      |_ dev5
   |_ features/

2. Set up 'HOW2_FEATS' and 'HOW_TEXT' in db.sh to where you put 'how-300h-v1'
   and 'fbank_pitch_181516' directories.

----

RECIPE NOTES:

Transcriptions are first generated from subtitles, text and audio are then
re-aligned using a Kaldi's GMM/HMM model trained on WSJ. Thus, the transcriptions
contains numbers, punctuations, symbols, ..., which is not best-suited
for character-based models.

Current recipe contains temporay normalization scripts and text replacements files.
Scripts were written based on replacement rules generated automatically with tools
not provided here.
It should be replaced soon by a more general perl script, similar to espnet1.

----

RESULTS:

## asr_train_rnn_extracted_bpe
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|44901|86.0|10.6|3.4|3.6|17.7|77.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|224373|93.3|2.9|3.8|3.3|10.1|77.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|66580|85.7|8.7|5.6|3.3|17.6|77.7|

## asr_train_rnn_extracted_char

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|44901|80.8|14.9|4.3|3.8|23.0|85.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_set_iwslt2019decode_asr_model_valid.acc.best|2497|224373|90.7|4.0|5.3|3.5|12.8|85.2|

```
