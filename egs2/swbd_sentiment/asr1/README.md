# RESULTS
## Dataset
- Speech Sentiment Annotations (Switchboard Sentiment)
   - Data: https://catalog.ldc.upenn.edu/LDC2020T14
   - Paper: https://catalog.ldc.upenn.edu/docs/LDC2020T14/LREC_2020_Switchboard_Senti.pdf

## Environments
- date: `Thu Mar  3 21:34:18 EST 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.9.0+cu102`
- Git hash: `3b53aedc654fd30a828689c2139a1e130adac077`
  - Commit date: `Fri Feb 25 00:13:16 2022 -0500`

## Using Conformer based encoder and Transformer based decoder with spectral augmentation and predicting transcript along with sentiment
- ASR config: [conf/tuning/train_asr_conformer.yaml](conf/tuning/train_asr_conformer.yaml)
- token_type: word
- labels: Positive, Neutral, Negative
- Pre-trained Model: https://huggingface.co/espnet/YushiUeda_swbd_sentiment_asr_train_asr_conformer

|dataset|Snt|Intent Classification Macro F1 (%)| Weighted F1 (%)| Micro F1 (%)|
|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/valid|2415|61.0|65.0|65.6|
|decode_asr_asr_model_valid.acc.ave_10best/test|2438|61.4|64.4|64.6|

## Using Conformer based encoder, Transformer based decoder and self-supervised learning features (Wav2vec2.0) with spectral augmentation and predicting transcript along with sentiment
- ASR config: [conf/tuning/train_asr_conformer_wav2vec2.yaml](conf/tuning/train_asr_conformer_wav2vec2.yaml)
- token_type: word
- labels: Positive, Neutral, Negative
- Pre-trained Model: https://huggingface.co/espnet/YushiUeda_swbd_sentiment_asr_train_asr_conformer_wav2vec2

|dataset|Snt|Intent Classification Macro F1 (%)| Weighted F1 (%)| Micro F1 (%)|
|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave_10best/valid|2415|64.5|67.5|67.4|
|decode_asr_asr_model_valid.acc.ave_10best/test|2438|64.1|66.5|66.3|
