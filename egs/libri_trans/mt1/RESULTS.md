# Transformer results

### train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_large_bpe8000 (sacreBLEU)

| dataset                                                                                                               | BLEU     | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------------------------------------------------------------------- | -------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_large_bpe8000/decode_dev.fr_decode_pytorch_transformer_bpe8k  | **24.8** | 53.4   | 29.0   | 18.7   | 13.0   | 1.000 | 1.005 | 22467   | 22365   |
| exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_large_bpe8000/decode_test.fr_decode_pytorch_transformer_bpe8k | **18.3** | 49.7   | 23.5   | 13.0   | 7.5    | 0.996 | 0.996 | 43664   | 43837   |

- Model files (archived to train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_large_bpe8000.tar.gz by `$ pack_model.sh`)
  - https://drive.google.com/file/d/1kod8WVLjsF9_5G_WpTAE0XNpdn1cJg5d/view?usp=sharing
  - training config file: `conf/tuning/train_pytorch_transformer_large.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe8k.yaml`
  - e2e file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_large_bpe8000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_large_bpe8000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000 (sacreBLEU)

| dataset                                                                                                         | BLEU     | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------------------------------------------------------------- | -------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/decode_dev.fr_decode_pytorch_transformer_bpe8k  | **21.5** | 52.2   | 26.7   | 15.9   | 9.9    | 0.992 | 0.992 | 22183   | 22365   |
| exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/decode_test.fr_decode_pytorch_transformer_bpe8k | **18.3** | 49.7   | 23.7   | 13.0   | 7.5    | 0.993 | 0.993 | 43552   | 43837   |

- Model files (archived to train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000.tar.gz by `$ pack_model.sh`)
  - https://drive.google.com/file/d/1W-1AxcpSneITalSIQPC-RuhTnZ7l6pjC/view?usp=sharing
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_bpe8k.yaml`
  - e2e file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe8000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000 (sacreBLEU)

| dataset                                                                                                   | BLEU     | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------------------------------------------------------- | -------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/decode_dev.fr_decode_pytorch_transformer  | **20.1** | 51.7   | 25.9   | 14.9   | 8.7    | 0.984 | 0.984 | 22007   | 22365   |
| exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/decode_test.fr_decode_pytorch_transformer | **18.3** | 49.8   | 23.6   | 13.0   | 7.4    | 0.995 | 0.995 | 43618   | 43837   |

- Model files (archived to train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1MY06WhBmg3Z5ZQKDduairTnqTFIVkDEd/view?usp=sharing
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
  - e2e file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.fr_lc.rm_tc_pytorch_train_pytorch_transformer_bpe1000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

# RNN baseline

### train.fr_lc.rm_lc_pytorch_train (character)

| dataset                                                   | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.fr_lc.rm_lc_pytorch_train/decode_dev.fr_decode  | **20.01** | 54.0   | 27.3   | 15.8   | 9.4    | 0.926 | 0.929 | 20860   | 22462   |
| exp/train.fr_lc.rm_lc_pytorch_train/decode_test.fr_decode | **18.39** | 52.6   | 25.6   | 14.3   | 8.1    | 0.926 | 0.929 | 40950   | 44080   |
