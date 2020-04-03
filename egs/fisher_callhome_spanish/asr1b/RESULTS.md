# Summary (WER)
|model|fisher_dev|fisher_dev2|fisher_test|callhome_devtest|callhome_evltest|
|-----|----------|-----------|-----------|----------------|----------------|
|RNN (char) [[Weiss et al.]](https://arxiv.org/abs/1703.08581)|25.7|25.1|23.2|44.5|45.3|
|RNN (BPE1k)|26.0|24.9|22.8|44.6|45.7|
|Transformer (BPE1k)|24.2|23.6|21.5|41.1|41.4|
| + SpecAugment|**23.1**|**22.5**|**20.8**|**40.2**|**39.6**|


# Transformer results
### train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_fisher_dev.es_decode_pytorch_transformer_bpe|3973|40966|81.0|12.0|7.0|4.1|**23.1**|65.8|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_fisher_dev2.es_decode_pytorch_transformer_bpe|3957|39895|81.9|11.9|6.1|4.5|**22.5**|65.6|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_fisher_test.es_decode_pytorch_transformer_bpe|3638|39990|84.1|10.8|5.2|4.9|**20.8**|64.7|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_callhome_devtest.es_decode_pytorch_transformer_bpe|3956|37584|67.4|22.7|9.9|7.6|**40.2**|80.3|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_callhome_evltest.es_decode_pytorch_transformer_bpe|1825|18807|67.4|22.8|9.8|7.0|**39.6**|82.0|

- Model files (archived to train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1CPo1a8r2OXi5zqDmqs3gbZQW9BB4KHsu
  - training config file: `conf/tuning/train_pytorch_transformer_bpe_long.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.es/cmvn.ark`
  - e2e file: `exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/results/model.json`
  - lm file: `exp/train_sp.es_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/rnnlm.model.best`
  - lm JSON file: `exp/train_sp.es_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_fisher_dev.es_decode_pytorch_transformer_bpe|3973|40966|80.0|12.8|7.2|4.2|**24.2**|66.3|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_fisher_dev2.es_decode_pytorch_transformer_bpe|3957|39895|81.0|12.6|6.4|4.5|**23.6**|65.8|s
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_fisher_test.es_decode_pytorch_transformer_bpe|3638|39990|83.5|11.0|5.5|5.0|**21.5**|65.7|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_callhome_devtest.es_decode_pytorch_transformer_bpe|3956|37584|66.2|23.5|10.2|7.3|**41.1**|80.7|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_callhome_evltest.es_decode_pytorch_transformer_bpe|1825|18807|65.7|23.7|10.6|7.1|**41.4**|82.7|


# RNN results
### train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_fisher_dev.es_decode_rnn_bpe|3973|40966|78.2|14.3|7.5|4.2|**26.0**|66.8|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_fisher_dev2.es_decode_rnn_bpe|3957|39895|79.5|14.2| 6.4|4.4|**24.9**|67.1|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_fisher_test.es_decode_rnn_bpe|3638|39990|82.0|12.4|5.5| 4.9|**22.8**|65.9|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_callhome_devtest.es_decode_rnn_bpe|3956|37584|61.7|26.1|12.2|6.3|**44.6**|81.5|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_callhome_evltest.es_decode_rnn_bpe|1825|18807|60.4|26.4|13.2|6.1|**45.7**|82.8|
