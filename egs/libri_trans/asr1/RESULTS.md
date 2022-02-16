# Conformer results

### train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug

| dataset                                                                                                         | Snt  | Wrd   | Corr | Sub | Del | Ins | Err     | S.Err |
| --------------------------------------------------------------------------------------------------------------- | ---- | ----- | ---- | --- | --- | --- | ------- | ----- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug/decode_dev.en_decode_pytorch_transformer  | 1071 | 18651 | 95.2 | 4.3 | 0.5 | 1.0 | **5.8** | 41.5  |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug/decode_test.en_decode_pytorch_transformer | 2048 | 36336 | 95.2 | 4.3 | 0.4 | 0.9 | **5.6** | 41.0  |

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1Ewj551fKYFWF6NsXXs4dzD9T_AeQMRjg/view?usp=sharing
  - training config file: `conf/tuning/train_pytorch_conformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug/results/model.json`
  - lm file: `exp/train_sp.en_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/rnnlm.model.best`
  - lm JSON file: `exp/train_sp.en_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

# Transformer results

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug

| dataset                                                                                                           | Snt  | Wrd   | Corr | Sub | Del | Ins | Err     | S.Err |
| ----------------------------------------------------------------------------------------------------------------- | ---- | ----- | ---- | --- | --- | --- | ------- | ----- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_dev.en_decode_pytorch_transformer  | 1071 | 18651 | 94.8 | 4.7 | 0.6 | 0.9 | **6.2** | 45.5  |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/decode_test.en_decode_pytorch_transformer | 2048 | 36336 | 94.3 | 5.1 | 0.6 | 0.9 | **6.6** | 47.0  |

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1gYhTYfN005PwLP4S8EHpt-MTTucSztZz/view?usp=sharing
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug/results/model.json`
  - lm file: `exp/train_sp.en_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/rnnlm.model.best`
  - lm JSON file: `exp/train_sp.en_lc.rm_rnnlm_pytorch_lm_lc.rm_bpe1000/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

# RNN results

### train_sp.en_lc.rm_pytorch_train_rnn_bpe1000

| dataset                                                                   | Snt  | Wrd   | Corr | Sub | Del | Ins | Err     | S.Err |
| ------------------------------------------------------------------------- | ---- | ----- | ---- | --- | --- | --- | ------- | ----- |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_dev.en_decode_rnn  | 1071 | 18651 | 93.4 | 6.0 | 0.6 | 1.1 | **7.7** | 54.3  |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_test.en_decode_rnn | 2048 | 36336 | 93.1 | 6.1 | 0.8 | 1.0 | **7.9** | 53.1  |
