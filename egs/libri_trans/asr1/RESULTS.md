# Transformer results
### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_dev.en_decode_pytorch_transformer_bpe|1071|18651|94.6|4.9|0.5|1.1|**6.5**|47.6|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/decode_test.en_decode_pytorch_transformer_bpe|2048|36336|94.6|4.9|0.5|1.0|**6.4**|45.9|

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1zH7Gx-rUvnVby6ww9--Hc7hMwO1dprqo
  - training config file: `conf/tuning/train_pytorch_transformer_bpe_long.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_long_bpe1000_specaug/results/model.json`
  - lm file: `exp/train_sp.en_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/rnnlm.model.best`
  - lm JSON file: `exp/train_sp.en_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_dev.en_decode_pytorch_transformer_bpe|1071|18651|93.7|5.6|0.7|1.4|**7.6**|51.3|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe_bpe1000/decode_test.en_decode_pytorch_transformer_bpe|2048|36336|93.7|5.6|0.7|1.2|**7.5**|49.2|

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe38
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe38/decode_dev.en_decode_pytorch_transformer|1071|18651|93.1|6.2|0.6|1.1|**8.0**|52.3|
|exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe38/decode_test.en_decode_pytorch_transformer|2048|36336|92.7|6.7|0.7|1.0|**8.4**|53.9|


# RNN results
### train_sp.en_lc.rm_pytorch_train_rnn_bpe_bpe1000
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_dev.en_decode_rnn_char|1071|18651|93.4|6.0|0.6|1.1|**7.7**|54.3|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe_bpe1000/decode_test.en_decode_rnn_char|2048|36336|93.1|6.1|0.8|1.0|**7.9**|53.1|

### train_sp.en_lc.rm_pytorch_train_bpe38
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_bpe38/decode_dev.en_decode_rnn_char|1071|18651|92.4|6.9|0.7|1.0|**8.6**|58.3|
|exp/train_sp.en_lc.rm_pytorch_train_bpe38/decode_test.en_decode_rnn_char|2048|36336|92.2|7.0|0.8|1.1|**8.9**|56.5|

### train_sp.en_lc.rm_pytorch_train_rnn_char
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_char/decode_dev.en_decode_rnn_char|1071|18651|91.3|7.8|0.8|1.0|**9.6**|62.7|
|exp/train_sp.en_lc.rm_pytorch_train_rnn_char/decode_test.en_decode_rnn_char|2048|36336|90.7|8.3|0.9|1.2|**10.4**|62.1|
