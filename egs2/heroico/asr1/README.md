# Heroico Recipe

This is the ASR recipe of the [LDC West Point Heroico Spanish Speech corpus](https://www.openslr.org/39/) (LDC2006S37, Apache 2.0 license).

The corpus consists of three subsets of Latin American Spanish speech:
- **Heroico Answers**: spontaneous answers to questions, recorded at the Military Academy of Heroico in Mexico
- **Heroico Recordings**: read speech of prompted sentences, recorded at the Military Academy of Heroico in Mexico
- **USMA**: read speech of prompted sentences, recorded at the United States Military Academy (West Point)

## How to run

```sh
bash run.sh
```

```
@misc{LDC2006S37,
	Author = {Morgan, John},
	Title = {West Point Heroico Spanish Speech},
	Publisher = {Linguistic Data Consortium},
	Address = {Philadelphia},
	Year = {2006},
	Note = {LDC Catalog No. LDC2006S37}}
```


# RESULTS
## Environments
- date: `Mon Jul  6 01:25:47 UTC 2026`
- python version: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.11.0+cu128`
- Git hash: `ff67a6d20f235f69e0a31446e00eef646a293d1d`
  - Commit date: `Wed Jun 24 23:09:10 2026 +0000`

## exp/asr_train_asr_conformer_raw_es_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_asr_model_valid.acc.ave/dev|1510|11447|90.9|6.9|2.2|1.1|10.2|28.9|
|decode_asr_transformer_asr_model_valid.acc.ave/test|4636|22206|86.2|12.3|1.5|2.0|15.8|37.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_asr_model_valid.acc.ave/dev|1510|64085|96.8|1.1|2.1|0.9|4.1|28.9|
|decode_asr_transformer_asr_model_valid.acc.ave/test|4636|116655|95.4|2.6|1.9|1.4|6.0|37.6|
