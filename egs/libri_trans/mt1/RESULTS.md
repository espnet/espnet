# Transformer results
### train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/decode_dev.fr_decode_pytorch_transformer_bpe|**19.64**|51.9|25.8|14.6|8.5|0.972|0.972|21791|22408|
|exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/decode_test.fr_decode_pytorch_transformer_bpe|**18.09**|50.0|23.8|13.1|7.4|0.982|0.982|43115|43904|

- Model files (archived to train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1W9sfhO2qHXSkSyr_qwAUWdagjkVg9KhF
  - training config file: `conf/tuning/train_pytorch_transformer_bpe.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe.yaml`
  - e2e file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)


# RNN baseline
### train.fr_lc.rm_lc_pytorch_train (character)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train.fr_lc.rm_lc_pytorch_train/decode_dev.fr_decode|**20.01**|54.0|27.3|15.8|9.4|0.926|0.929|20860|22462|
|exp/train.fr_lc.rm_lc_pytorch_train/decode_test.fr_decode|**18.39**|52.6|25.6|14.3|8.1|0.926|0.929|40950|44080|
