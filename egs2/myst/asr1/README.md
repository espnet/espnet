# MyST RECIPE

This is the recipe of the children speech recognition model with [MyST dataset](https://catalog.ldc.upenn.edu/LDC2021S05).

Before running the recipe, please download from https://catalog.ldc.upenn.edu/LDC2021S05.
Then, edit 'MYST' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
MYST=/path/to/myst

$ tree -L 2 /path/to/myst
/path/to/myst
└── myst_child_conv_speech
    ├── data
    ├── docs
    └── index.html
```


# RESULTS

## exp/asr_asr_train_asr_wavlm_transformer_raw_en_bpe5000_sp_bs16000000

Model: https://huggingface.co/espnet/myst_wavlm_aed_transformer

## Environments
- date: `Mon Nov 25 21:12:07 CST 2024`
- python version: `3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]`
- espnet version: `espnet 202409`
- pytorch version: `pytorch 2.4.0`
- Git hash: `6b5c6230a794aa4a5df872be69e417a3fbfe821b`
  - Commit date: `Sun Nov 24 23:13:48 2024 -0600`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|13180|202306|88.4|7.6|4.0|3.4|15.0|61.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|13180|1016043|93.2|2.1|4.7|3.6|10.4|61.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|13180|228240|86.4|6.7|6.8|4.0|17.6|61.9|


# References
[1] Pradhan, Sameer, Ronald Cole, and Wayne Ward. "My Science Tutor (MyST)–a Large Corpus of Children’s Conversational Speech." Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.
