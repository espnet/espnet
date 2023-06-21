# Conformer + specaug + speed perturbation
## Environments
 - date: `Thu Jan 13 17:16:42 CST 2022`
 - python version: `3.8.12 (default, Oct 12 2021, 13:49:34)  [GCC 7.5.0]`
 - espnet version: `espnet 0.10.6a1`
 - pytorch version: `pytorch 1.8.1+cu111`
 - Git hash: `cddeeef1933ce4c1552e9d2e1af5bb3c60ad74f4`
   - Commit date: `Fri Dec 31 23:16:25 2021 +0900`

## With Transformer LM
 - Model link: [zenodo](https://zenodo.org/record/5845307) / [huggingface](https://huggingface.co/espnet/vectominist_seame_asr_conformer_bpe5626)
 - ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)
 - LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)
 
### WER
 Mixed Mandarin CER / English WER
 |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
 |---|---|---|---|---|---|---|---|---|
 |decode_lm0.2_ctc0.4_beam10/devman|6531|96737|85.3|11.4|3.3|1.9|16.6|75.5|
 |decode_lm0.2_ctc0.4_beam10/devsge|5321|54390|79.5|16.2|4.4|2.8|23.3|74.3|

 Mandarin CER
 |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
 |---|---|---|---|---|---|---|---|---|
 |decode_lm0.2_ctc0.4_beam10/devman|6531|71806|88.2|7.6|4.3|3.2|15.0|59.4|
 |decode_lm0.2_ctc0.4_beam10/devsge|5321|20327|84.9|9.1|6.0|6.9|22.0|34.5|

 English WER
 |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
 |---|---|---|---|---|---|---|---|---|
 |decode_lm0.2_ctc0.4_beam10/devman|6531|24931|76.9|14.4|8.7|6.1|29.2|52.6|
 |decode_lm0.2_ctc0.4_beam10/devsge|5321|34063|76.2|16.2|7.5|4.5|28.2|66.2|