# RESULTS
## Dataset
- IEMOCAP database: The Interactive Emotional Dyadic Motion Capture database
  - Database: https://sail.usc.edu/iemocap/
  - Paper: https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf

## Environments
- date: `Thu Sep  9 21:55:50 EDT 2021`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.9.0+cu102`
- Git hash: `5dacbea87269472b60c5c24d42a09900f81a27c7`
  - Commit date: `Wed Sep 8 15:23:33 2021 +0900`


## Using Transformer based encoder-decoder and self-supervised learning features [HuBERT_large_ll60k, Transformer, utt_mvn](conf/tuning/train_asr_transformer_hubert_960hr_large.yaml) and predicting transcript along with emotion
- ASR config: [conf/tuning/train_asr_transformer_hubert_960hr_large.yaml](conf/tuning/train_asr_transformer_hubert_960hr_large.yaml)
- token_type: word
- keep_nbest_models: 10
- Emotional Labels: anger, happiness, sadness and neutral

|dataset|Snt|Emotion Classification (%)|
|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/test|941|69.53|
|decode_asr_asr_model_valid.acc.ave_10best/valid|390|77.18|

### ASR results

#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/test|941|11017|75.7|15.1|9.2|5.6|29.9|76.1|
|decode_asr_asr_model_valid.acc.ave_10best/valid|390|4355|82.8|9.4|7.9|3.3|20.5|58.5|

