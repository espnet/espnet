# Summary (WER)
|model|fisher_dev|fisher_dev2|fisher_test|callhome_devtest|callhome_evltest|
|-----|----------|-----------|-----------|----------------|----------------|
|RNN (char) [[Weiss et al.]](https://arxiv.org/abs/1703.08581)|25.7|25.1|23.2|44.5|45.3|
|RNN (BPE1k)|26.0|24.9|22.8|44.6|45.7|
|Transformer (BPE1k)|24.3|23.6|21.5|40.8|41.2|
|+ SpecAugment|**23.0**|**22.3**|**20.6**|**39.6**|**39.3**|

### RNN (BPE1k)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_fisher_dev.es_decode_rnn_bpe|3973|40966|78.2|14.3|7.5|4.2|**26.0**|66.8|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_fisher_dev2.es_decode_rnn_bpe|3957|39895|79.5|14.2| 6.4|4.4|**24.9**|67.1|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_fisher_test.es_decode_rnn_bpe|3638|39990|82.0|12.4|5.5| 4.9|**22.8**|65.9|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_callhome_devtest.es_decode_rnn_bpe|3956|37584|61.7|26.1|12.2|6.3|**44.6**|81.5|
|exp/train_sp.es_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_callhome_evltest.es_decode_rnn_bpe|1825|18807|60.4|26.4|13.2|6.1|**45.7**|82.8|

### Transformer (BPE1k)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_fisher_dev.es_decode_pytorch_transformer_bpe|3973|40966|80.1|12.9|7.1|4.4|**24.3**|66.2|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_fisher_dev2.es_decode_pytorch_transformer_bpe|3957|39895|81.2|12.6|6.3|4.8|**23.6**|66.0|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_fisher_test.es_decode_pytorch_transformer_bpe|3638|39990|83.6|11.2|5.2|5.1|**21.5**|65.1|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_callhome_devtest.es_decode_pytorch_transformer_bpe|3956|37584|66.6|23.5|9.9|7.4|**40.8**|80.7|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_callhome_evltest.es_decode_pytorch_transformer_bpe|1825|18807|66.0|24.0|10.0|7.2|**41.2**|81.8|

- Model files (archived to train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1kYHrZILYpKNjHPdJk9ZxuLMk9JT0nEHH
  - training config file: `conf/tuning/train_pytorch_transformer_bpe.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - cmvn file: `data/train_sp.es/cmvn.ark`
  - e2e file: `exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/results/model.json`
  - lm file: `exp/train_sp.es_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/rnnlm.model.best`
  - lm JSON file: `exp/train_sp.es_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### Transformer (BPE1k) + SpecAugment
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_fisher_dev.es_decode_pytorch_transformer_bpe|3973|40966|80.9|12.1|7.0|4.0|**23.0**|65.4|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_fisher_dev2.es_decode_pytorch_transformer_bpe|3957|39895|82.1|11.7|6.1|4.5|**22.3**|65.0|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_fisher_test.es_decode_pytorch_transformer_bpe|3638|39990|84.3|10.5|5.2|4.9|**20.6**|64.7|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_callhome_devtest.es_decode_pytorch_transformer_bpe|3956|37584|67.9|23.0|9.1|7.4|**39.6**|80.1|
|exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_callhome_evltest.es_decode_pytorch_transformer_bpe|1825|18807|67.8|22.8|9.4|7.2|**39.3**|81.3|

- Model files (archived to train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1ECjnvjX2al0f0JirE2G-eh6ZYew2T7B9
  - training config file: `conf/tuning/train_pytorch_transformer_bpe_long.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.es/cmvn.ark`
  - e2e file: `exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.es_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/results/model.json`
  - lm file: `exp/train_sp.es_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/rnnlm.model.best`
  - lm JSON file: `exp/train_sp.es_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
