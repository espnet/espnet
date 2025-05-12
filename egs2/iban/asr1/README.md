# Iban Recipe
This is the ASR recipe of the [Iban text and speech copora](https://www.openslr.org/24/).

```
@inproceedings{Juan14,
	Author = {Sarah Samson Juan and Laurent Besacier and Solange Rossato},
	Booktitle = {Proceedings of Workshop for Spoken Language Technology for Under-resourced (SLTU)},
	Month = {May},
	Title = {Semi-supervised G2P bootstrapping and its application to ASR for a very under-resourced language: Iban},
	Year = {2014}}

@inproceedings{Juan2015,
  	Title = {Using Resources from a closely-Related language to develop ASR for a very under-resourced Language: A case study for Iban},
  	Author = {Sarah Samson Juan and Laurent Besacier and Benjamin Lecouteux and Mohamed Dyab},
  	Booktitle = {Proceedings of INTERSPEECH},
  	Year = {2015},
  	Address = {Dresden, Germany},
  	Month = {September}}
```


# RESULTS
## Environments
- date: `Mon May 12 12:37:16 EDT 2025`
- python version: `3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]`
- espnet version: `espnet 202412`
- pytorch version: `pytorch 2.0.1`
- Git hash: `9e12b0c877d28fba8ae1ce71abf6ed91c05d9238`
  - Commit date: `Tue May 6 07:28:58 2025 -0400`
- GPU: 1 V100-32GB
- Model: https://huggingface.co/cjli/iban_wavlm_conformer

## exp/asr_train_asr_wavlm_conformer_raw_iba_bpe200_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|104|2226|71.2|23.8|5.1|2.3|31.2|94.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|104|13527|91.8|2.7|5.4|2.2|10.4|94.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|104|5758|81.2|10.9|7.8|2.7|21.4|94.2|

## exp/asr_train_asr_wavlm_conformer_raw_iba_bpe200_sp/decode_asr_asr_model_valid.acc.best
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|473|11006|80.0|16.4|3.5|2.4|22.3|92.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|473|67025|94.9|1.9|3.3|1.7|6.9|92.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|473|27176|87.7|7.5|4.8|1.9|14.2|92.0|
