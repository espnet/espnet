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

# Sentiment Analysis RESULTS
## Environments
- date: `Thu Feb 17 11:25:22 EST 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.9.0+cu102`
- Git hash: `f6cde1c419c814a14ccd40abe557a780508cbcdf`
  - Commit date: `Fri Feb 11 12:25:33 2022 -0500`

## Using Conformer based encoder and Transformer based decoder with spectral augmentation and predicting transcript along with sentiment
- ASR config: [conf/tuning/train_asr_conformer.yaml](conf/tuning/train_asr_conformer.yaml)
- token_type: word
- Sentiment Labels: Positive, Neutral, Negative
- Pretrained Model
   - Hugging Face: https://huggingface.co/espnet/YushiUeda_iemocap_sentiment_asr_train_asr_conformer

|dataset|Snt|Intent Classification Macro F1 (%)| Weighted F1 (%)| Micro F1 (%)|
|---|---|---|---|---|
|decode_asr_model_valid.acc.ave_10best/valid|754|53.9|65.7|66.4|
|decode_asr_model_valid.acc.ave_10best/test|1650|50.3|54.5|55.7|

## Using Conformer based encoder, Transformer based decoder, and self-supervised learning features with spectral augmentation and predicting transcript along with sentiment
- ASR config: [conf/tuning/train_asr_conformer_hubert.yaml](conf/tuning/train_asr_conformer_hubert.yaml)
- token_type: word
- Sentiment Labels: Positive, Neutral, Negative
- Pretrained Model
   - Hugging Face: https://huggingface.co/espnet/YushiUeda_iemocap_sentiment_asr_train_asr_conformer_hubert

|dataset|Snt|Intent Classification Macro F1 (%)| Weighted F1 (%)| Micro F1 (%)|
|---|---|---|---|---|
|decode_asr_model_valid.acc.ave_10best/valid|754|66.5|76.4|75.7|
|decode_asr_model_valid.acc.ave_10best/test|1650|62.0|65.5|65.8|
