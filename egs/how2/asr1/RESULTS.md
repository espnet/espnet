# Transformer results
### train.en_lc.rm_pytorch_train_bpe5000
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train.en_lc.rm_pytorch_train_bpe5000/decode_dev5.en_decode|2305|42298|90.4|7.2|2.4|3.4|**13.0**|68.6|
|exp/train.en_lc.rm_pytorch_train_bpe5000/decode_test_set_iwslt2019.en_decode|2497|46505|90.4|7.0|2.6|3.6|**13.1**|69.6|

- Model files (archived to train.en_lc.rm_pytorch_train_bpe5000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1pV6IS2DlJWf_fo1YdINl_FfubqZpUhNw
  - training config file: `conf/train.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train.en/cmvn.ark`
  - e2e file: `exp/train.en_lc.rm_pytorch_train_bpe5000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.en_lc.rm_pytorch_train_bpe5000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)


# RNN results
### train.en_lc.rm_pytorch_train_bpe1000
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train.en_lc.rm_pytorch_train_bpe1000/decode_dev5.en_decode/|2305|42298|88.6|8.7|2.7|3.2|**14.6**|70.8|
|exp/train.en_lc.rm_pytorch_train_bpe1000/decode_test_set_iwslt2019.en_decode/|2497|46505|88.7|8.6|2.8|3.3|**14.6**|73.4|
