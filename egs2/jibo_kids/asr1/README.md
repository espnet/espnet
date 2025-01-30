# JIBO Kids RECIPE

This is the recipe of the children speech recognition model with [JIBO Kids dataset](https://github.com/balaji1312/Jibo_Kids).

Before running the recipe, please download the [dataset](https://github.com/balaji1312/Jibo_Kids).
Then, edit 'JIBO_KIDS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
JIBO_KIDS=/path/to/jibo_kids

$ tree -L 2 path/to/jibo_kids
/path/to/jibo_kids
├── data
│   ├── blocks
│   ├── brush
│   ├── colors
│   └── letters_digits
└── README.txt

```

# RESULTS

Model: https://huggingface.co/balaji1312/jibo_kids_wavlm_aed_transformer

## Environments
- date: `Thu Jan 30 06:18:01 EST 2025`
- python version: `3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.4.0`
- Git hash: `c46aa9a594ff83d52cbf61d84c5650325d1ce527`
  - Commit date: `Sun Oct 13 14:39:31 2024 -0400`

## exp/asr_train_asr_wavlm_transformer_raw_en_bpe1024
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|1044|3686|56.1|31.4|12.5|8.1|52.0|62.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|1044|16215|75.4|8.1|16.6|9.4|34.1|62.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|1044|5220|64.5|18.0|17.5|10.4|45.9|62.3|

## exp/asr_train_asr_wavlm_transformer_raw_en_bpe1024/decode_asr_asr_model_valid.acc.best
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|853|2372|59.8|31.2|8.9|7.2|47.3|64.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|853|9855|78.3|7.3|14.3|8.4|30.1|64.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|853|3590|68.2|16.2|15.6|6.4|38.3|64.0|

## References

[1] Shankar, Natarajan Balaji, et al. "The JIBO Kids Corpus: A speech dataset of child-robot interactions in a classroom environment." JASA Express Letters 4.11 (2024).
