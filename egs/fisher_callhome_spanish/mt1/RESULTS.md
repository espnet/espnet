# NOTE: apostrophe is included both in hyp and ref

# Summary (4-gram BLEU)

| model                                                         | fisher_dev | fisher_dev2 | fisher_test | callhome_devtest | callhome_evltest |
| ------------------------------------------------------------- | ---------- | ----------- | ----------- | ---------------- | ---------------- |
| RNN (char) [[Weiss et al.]](https://arxiv.org/abs/1703.08581) | 58.70      | 59.90       | 57.90       | 28.20            | 27.90            |
| Transformer (char)                                            | **62.90**  | **64.31**   | **61.64**   | **31.54**        | **31.62**        |
| Transformer (BPE1k)                                           | 62.59      | 63.22       | 61.45       | 30.58            | 29.86            |

# Transformer results

### train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000

| dataset                                                                                                                  | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------------------------------------------------------------ | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/decode_fisher_dev.en_decode_pytorch_transformer       | **62.59** | 88.0   | 71.0   | 56.1   | 43.8   | 1.000 | 1.001 | 39799   | 39772   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/decode_fisher_dev2.en_decode_pytorch_transformer      | **63.22** | 88.2   | 71.6   | 57.0   | 44.6   | 0.998 | 0.998 | 38815   | 38877   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/decode_fisher_test.en_decode_pytorch_transformer      | **61.45** | 87.9   | 70.1   | 54.8   | 42.2   | 1.000 | 1.002 | 38852   | 38761   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/decode_callhome_devtest.en_decode_pytorch_transformer | **30.58** | 61.6   | 37.6   | 24.2   | 16.0   | 0.994 | 0.994 | 37198   | 37416   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/decode_callhome_evltest.en_decode_pytorch_transformer | **29.86** | 60.0   | 36.2   | 23.6   | 15.6   | 0.999 | 0.999 | 18435   | 18457   |

- Model files (archived to train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1nScq_ZU0vGgPixwyXP9cUgGlPVC-VByd
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
  - e2e file: `exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/results/model.val5.avg.best`
  - e2e JSON file: `exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_bpe1000/results/model.json`
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_char_bpe53

| dataset                                                                                                                          | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| -------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_char_bpe53/decode_fisher_dev.en_decode_pytorch_transformer_char       | **62.90** | 88.9   | 72.6   | 58.0   | 45.5   | 0.979 | 0.979 | 39095   | 39926   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_char_bpe53/decode_fisher_dev2.en_decode_pytorch_transformer_char      | **64.31** | 89.3   | 73.2   | 58.9   | 46.5   | 0.988 | 0.989 | 38505   | 38952   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_char_bpe53/decode_fisher_test.en_decode_pytorch_transformer_char      | **61.64** | 88.8   | 71.5   | 56.3   | 43.6   | 0.981 | 0.982 | 38234   | 38954   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_char_bpe53/decode_callhome_devtest.en_decode_pytorch_transformer_char | **31.54** | 62.6   | 38.9   | 25.7   | 17.5   | 0.975 | 0.975 | 36496   | 37416   |
| exp/train.en_lc.rm_lc.rm_pytorch_train_pytorch_transformer_char_bpe53/decode_callhome_evltest.en_decode_pytorch_transformer_char | **31.62** | 61.5   | 38.3   | 25.3   | 17.1   | 0.994 | 0.994 | 18354   | 18457   |

- NOTE: this is quite slow

# RNN results

### train.en_lc.rm_lc_pytorch_train

| dataset                                                                                        | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ---------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train.en_lc.rm_lc_pytorch_train/decode_fisher_dev.en_decode_pytorch_transformer_char       | **60.68** | 86.3   | 69.2   | 54.3   | 41.8   | 1.000 | 1.015 | 40791   | 40196   |
| exp/train.en_lc.rm_lc_pytorch_train/decode_fisher_dev2.en_decode_pytorch_transformer_char      | **62.05** | 87.3   | 70.5   | 55.7   | 43.2   | 1.000 | 1.009 | 39726   | 39360   |
| exp/train.en_lc.rm_lc_pytorch_train/decode_fisher_test.en_decode_pytorch_transformer_char      | **59.63** | 86.5   | 68.6   | 53.0   | 40.2   | 1.000 | 1.019 | 39922   | 39186   |
| exp/train.en_lc.rm_lc_pytorch_train/decode_callhome_devtest.en_decode_pytorch_transformer_char | **29.46** | 60.2   | 36.2   | 22.9   | 15.1   | 1.000 | 1.020 | 38168   | 37424   |
| exp/train.en_lc.rm_lc_pytorch_train/decode_callhome_evltest.en_decode_pytorch_transformer_char | **28.97** | 58.4   | 35.1   | 22.7   | 15.2   | 1.000 | 1.036 | 19129   | 18463   |
