# Branchformer: init

- ASR config: [conf/tuning/train_asr_branchformer_e18_d6_size512_lr1e-3_warmup35k.yaml](conf/tuning/train_asr_branchformer_e18_d6_size512_lr1e-3_warmup35k.yaml)
- #Params: 95.64 M
- Model link: [https://huggingface.co/pyf98/slurp_entity_branchformer](https://huggingface.co/pyf98/slurp_entity_branchformer)

## Environments
- date: `Fri May 27 03:41:59 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.11.0`
- Git hash: `4f36236ed7c8a25c2f869e518614e1ad4a8b50d6`
  - Commit date: `Thu May 26 00:22:45 2022 -0400`

# Intent Classification
- Valid Intent Classification Result: 0.8727272727272728
- Test Intent Classification Result: 0.8653463832390274

# Entity
|Slu f1|Precision|Recall|F-Measure|
|:---:|:---:|:---:|:---:|
|test|0.7972|0.7564|0.7763|



# Conformer: new config

- ASR config: [conf/tuning/train_asr_conformer_e12_d6_size512_lr1e-3_warmup35k.yaml](conf/tuning/train_asr_conformer_e12_d6_size512_lr1e-3_warmup35k.yaml)
- #Params: 109.39 M
- Model link: [https://huggingface.co/pyf98/slurp_entity_conformer](https://huggingface.co/pyf98/slurp_entity_conformer)

## Environments
- date: `Thu May 26 14:51:29 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.11.0`
- Git hash: `4f36236ed7c8a25c2f869e518614e1ad4a8b50d6`
  - Commit date: `Thu May 26 00:22:45 2022 -0400`

## Intent Classification
- Valid Intent Classification Result: 0.8678941311852704
- Test Intent Classification Result: 0.8652699189478513

## Entity
|Slu f1|Precision|Recall|F-Measure|
|:---:|:---:|:---:|:---:|
|test|0.7956|0.7443|0.7691|



# Using XLS-R pretrained speech Encoder and mBART-50 Large pretrained text Encoder-Decoder

- ASR config: [conf/tuning/train_asr_branchformer_xlsr_mbart.yaml](conf/tuning/train_asr_branchformer_xlsr_mbart.yaml)
- #Params: 1.21 B

## Environments
- date: `Wed Sep  7 01:16:08 CEST 2022`
- python version: `3.9.13 (main, Jun  9 2022, 00:00:00)  [GCC 11.3.1 20220421 (Red Hat 11.3.1-2)]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.12.1+cu116`
- Git hash: `c9cb7c424c90e9d3a59ace324308793b91fedbe1`
- Commit date: `Tue Aug 23 16:22:24 2022 +0200`

## Intent Classification
- Valid Intent Classification Result: 0.8933256616800921
- Test Intent Classification Result: 0.8811744915124636

## Entity
|Slu f1|Precision|Recall|F-Measure|
|:---:|:---:|:---:|:---:|
|test|0.7949|0.7788|0.7868|

# Initial Result

## Environments
- date: `Thu Oct 28 16:54:32 2021 -0400`
- python version: `3.9.5 (default, Jun  4 2021, 12:28:51) [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `d7093719d98692774bb47d3c9470a1ca94d33866`
  - Commit date: `Thu Oct 28 16:54:32 2021 -0400`

## Using Conformer based encoder and Transformer based decoder with spectral augmentation and predicting transcript along with intent
- ASR config: [conf/train_asr.yaml](conf/tuning/train_asr_conformer.yaml)
- token_type: word
- Entity classification code borrowed from SLURP [1] official repo - https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation
- Pretrained Model
  - Zenodo : https://zenodo.org/record/5651224
  - Hugging Face : https://huggingface.co/espnet/siddhana_slurp_entity_asr_train_asr_conformer_raw_en_word_valid.acc.ave_10best

|dataset|Snt|Entity Classification (F1 Score)|
|---|---|---|
|inference_asr_model_valid.acc.ave_10best/test|13078|71.9|

### Intent Classification Results


|dataset|Snt|Intent Classification (%)|
|---|---|---|
|inference_asr_model_valid.acc.ave_10best/test|13078|84.4|
|inference_asr_model_valid.acc.ave_10best/valid|8690|85.4|


## Citation

```
@inproceedings{slurp,
    author = {Emanuele Bastianelli and Andrea Vanzo and Pawel Swietojanski and Verena Rieser},
    title={{SLURP: A Spoken Language Understanding Resource Package}},
    booktitle={{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)}},
    year={2020}
}
```
