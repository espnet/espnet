# Corpus
**Free ST Chinese Mandarin Corpus**: a free Mandarin Chinese corpus collected by Surfingtech (www.surfing.ai). The dataset contains 102600 utterances from 855 speakers, for a total of 109.73 hours of speech. 

Since all speakers have 120 utterances, we manually divide the data into train, dev, and test split with a ratio of 90-5-5 using speaker IDs, resulting in 769, 43, and 43 speakers in our train, dev, test split respectively. Utterances with the same speaker ID are kept in the same split.

The original dataset contains duplicates sentences with the same transcript, but are spoken by different speakers. Although the waveforms are different for these duplicates, we still remove sentences in the test and development set that have duplicate transcripts in the training set, in order to eliminate any effect of training data leakage.

Link: https://openslr.org/38

# Results
## Environments
- python version: `3.9.10 | packaged by conda-forge | (main, Feb  1 2022, 21:24:11)  [GCC 9.4.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1`
- pretrained model: https://huggingface.co/espnet/zh_openslr38/blob/main/exp/asr_train_asr_conformer_raw_zh_char_sp/valid.acc.ave_10best.pth

## Spectrum Features
Code to reproduce:
```./run.sh```

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|4322|46490|91.0|8.4|0.5|0.2|9.2|51.5|
|decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|4167|45803|91.1|8.5|0.5|0.2|9.1|52.2|

## HuBERT Self-Supervised Learning (SSLR)
We provide the script to train with SSLR features via HuBERT. Due to the much longer training time with HuBERT, we only train for 24 epochs. The model does not show a lower CER over spectrum features, but training for longer may lead to improved results.

Code to reproduce:
```./local/run_sslr.sh```

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.acc.best/dev|4322|46490|90.8|8.6|0.6|0.2|9.4|51.9|
|decode_asr_lm_lm_train_lm_zh_char_valid.loss.ave_asr_model_valid.acc.best/test|4167|45803|90.8|8.7|0.5|0.2|9.4|54.1|
