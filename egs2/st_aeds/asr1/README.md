# RESULTS

This is an ESPnet2 ASR recipe for the Free ST American English Corpus
(ST-AEDS), OpenSLR SLR45.

- Dataset page: https://www.openslr.org/45/
- Archive: `ST-AEDS-20180100_1-OS.tgz`
- License: Creative Commons BY-NC-ND 4.0

## Environments

- date: `Wed Aug  5 17:35:32 EDT 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:43:48) [Clang 20.1.8 ]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.12.0`
- Git hash: `427a363b4ea39e854985230efa68a94deb196716`
  - Commit date: `Wed Jul 1 11:01:30 2026 -0400`

## Results

- ASR config: [conf/tuning/train_asr_rnn.yaml](conf/tuning/train_asr_rnn.yaml)
- Decode config: [conf/tuning/decode_rnn.yaml](conf/tuning/decode_rnn.yaml)

The baseline is a small character-level RNN CTC/attention ASR setup based on
`egs2/timit/asr1/conf/tuning/train_asr_rnn.yaml`.

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|decode_asr_asr_model_valid.acc.ave/org/dev|400|3071|53.2|43.1|3.7|7.6|54.4|98.5|
|decode_asr_asr_model_valid.acc.ave/test|400|3103|54.7|42.6|2.6|8.9|54.1|98.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|decode_asr_asr_model_valid.acc.ave/org/dev|400|16306|82.7|9.7|7.7|5.1|22.4|98.5|
|decode_asr_asr_model_valid.acc.ave/test|400|16654|83.2|9.6|7.2|4.8|21.6|98.8|

## ASR config

```sh
./run.sh \
    --lang en \
    --audio_format wav \
    --feats_type raw \
    --token_type char \
    --use_lm false \
    --asr_config conf/train_asr.yaml \
    --inference_config conf/decode_asr.yaml
```

## Data preparation

Set `ST_AEDS` in `db.sh` or pass it in the environment. The default value from
`../../TEMPLATE/asr1/db.sh` is `downloads`.

```sh
cd egs2/st_aeds/asr1
./local/data.sh
```

The preparation script creates `data/train`, `data/dev`, and `data/test`.
For each speaker, sorted utterances are split deterministically with 40
utterances for dev, 40 utterances for test, and the rest for train.

Current usable split sizes:

|dataset|utterances|
|---|---:|
|train|3038|
|dev|400|
|test|400|

Transcript rows with missing audio or empty transcript text are skipped.

The dataset license includes non-commercial and no-derivatives terms.
