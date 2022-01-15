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
 (Mandarin CER / English WER)
 |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
 |---|---|---|---|---|---|---|---|---|
 |decode_lm0.2_ctc0.4_beam30/devman|6531|96737|85.2|11.3|3.5|1.8|16.5|75.5|
 |decode_lm0.2_ctc0.4_beam30/devsge|5321|54390|79.4|16.0|4.6|2.6|23.2|74.3|

