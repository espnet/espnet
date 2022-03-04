# Results

## mt_train_mt_transformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3_raw_bpe_tc10000
- mt_config: conf/tuning/train_mt_transformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3.yaml
- inference_config: conf/decode_mt.yaml

### BLEU

Metric: BLEU-4, detokenized case-sensitive BLEU result (single-reference)

|dataset|bleu_score|verbose_score|
|---|---|---|
|beam5_maxlenratio1.6_penalty0.2/valid|33.3|68.4/42.9/28.9/19.8 (BP = 0.924 ratio = 0.927 hyp_len = 134328 ref_len = 144976)|
|beam5_maxlenratio1.6_penalty0.2/test|32.2|67.2/41.4/27.4/18.5 (BP = 0.933 ratio = 0.935 hyp_len = 119813 ref_len = 128122)|
