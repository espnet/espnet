# Transformer results
### train.pt_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/decode_dev5.pt_decode_pytorch_transformer.lc.rm|**54.25**|78.4|60.3|48.0|38.4|0.998|0.998|43983|44062|

- Model files (archived to train.pt_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=148p6-_wj7bcruxcw2qsg9cNKZu8DLhtY
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.lc.rm.yaml`
  - e2e file: `exp/train.pt_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.pt_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train.pt_tc_tc_pytorch_train_pytorch_transformer_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_tc_pytorch_train_pytorch_transformer_bpe8000/decode_dev5.pt_decode_pytorch_transformer.tc|**58.61**|80.7|64.1|52.6|43.3|1.000|1.001|44112|44062|

- Model files (archived to train.pt_tc_tc_pytorch_train_pytorch_transformer_bpe8000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1-dyzVeBDlFQ5zp9h-QODhmvHemTzU_nm
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.tc.yaml`
  - e2e file: `exp/train.pt_tc_tc_pytorch_train_pytorch_transformer_bpe8000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.pt_tc_tc_pytorch_train_pytorch_transformer_bpe8000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)


# RNN results
### train.pt_tc_tc_pytorch_train_bpe8000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.pt_tc_tc_pytorch_train_bpe8000/decode_dev5.pt_decode|**54.02**|78.2|60.1|47.8|38.1|0.999|0.999|44004|44062|
