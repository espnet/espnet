# Branchformer
- MT config: [conf/tuning/train_mt_branchformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3.yaml](conf/tuning/train_mt_branchformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3.yaml)
- #Params: 43.40 M
- Model link: [https://huggingface.co/pyf98/iwslt14_de_en_branchformer](https://huggingface.co/pyf98/iwslt14_de_en_branchformer)

## Environments
- date: `Sun May 29 01:39:59 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: `1cab3306f8136e614339390f59f06e11d054bbd8`
  - Commit date: `Sat May 28 20:15:14 2022 -0400`

## mt_train_mt_branchformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3_raw_bpe_tc10000
### BLEU

|dataset|score|verbose_score|
|---|---|---|
|beam5_maxlenratio1.6_penalty0.4/valid|34.1|67.6/42.9/29.0/20.0 (BP = 0.945 ratio = 0.946 hyp_len = 137200 ref_len = 144976)|
|beam5_maxlenratio1.6_penalty0.4/test|32.7|66.9/41.5/27.5/18.7 (BP = 0.945 ratio = 0.946 hyp_len = 121209 ref_len = 128122)|



# Transformer

## mt_train_mt_transformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3_raw_bpe_tc10000
- mt_config: conf/tuning/train_mt_transformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3.yaml
- inference_config: conf/decode_mt.yaml

### BLEU
Metric: BLEU-4, detokenized case-sensitive BLEU result (single-reference)

|dataset|bleu_score|verbose_score|
|---|---|---|
|beam5_maxlenratio1.6_penalty0.2/valid|33.3|68.4/42.9/28.9/19.8 (BP = 0.924 ratio = 0.927 hyp_len = 134328 ref_len = 144976)|
|beam5_maxlenratio1.6_penalty0.2/test|32.2|67.2/41.4/27.4/18.5 (BP = 0.933 ratio = 0.935 hyp_len = 119813 ref_len = 128122)|
