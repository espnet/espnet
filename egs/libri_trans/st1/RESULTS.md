# Transformer results

### ensemble (1) + (2) + (3)

| dataset                                                                                                                                            | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_dev.fr_decode_pytorch_transformer_pretrain_ensemble3  | **19.17** | 51.3   | 25.0   | 14.1   | 8.2    | 0.974 | 0.975 | 21842   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_test.fr_decode_pytorch_transformer_pretrain_ensemble3 | **17.40** | 49.8   | 23.4   | 12.7   | 7.0    | 0.972 | 0.972 | 42676   | 43904   |

### train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000_specaug_asrtrans_mttrans

| dataset                                                                                                                                                  | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000_specaug_asrtrans_mttrans/decode_dev.fr_decode_pytorch_transformer_pretrain  | **18.20** | 50.5   | 24.1   | 13.3   | 7.6    | 0.971 | 0.971 | 21760   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000_specaug_asrtrans_mttrans/decode_test.fr_decode_pytorch_transformer_pretrain | **16.76** | 48.6   | 22.3   | 12.0   | 6.7    | 0.976 | 0.976 | 42850   | 43904   |

### train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans (1)

| dataset                                                                                                                                  | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ---------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_dev.fr_decode_pytorch_transformer_pretrain  | **18.07** | 50.4   | 23.9   | 13.2   | 7.5    | 0.974 | 0.974 | 21823   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_test.fr_decode_pytorch_transformer_pretrain | **16.70** | 48.9   | 22.4   | 11.9   | 6.6    | 0.976 | 0.976 | 42845   | 43904   |

- Model files (archived to train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1n9LOCN-k_3HMH6uawe440zAntDBDf16N
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_pretrain.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.fr/cmvn.ark`
  - e2e file: `exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/results/model.json`
  - NOTE: This model is initialized with the Transformer ASR model (BPE1k, use SpecAugment) on the encoder side and Transformer MT model (BPE1k) on the decoder side.
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
- NOTE: longer version of "short" for SpecAugment: 30ep->50ep

### train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans (2)

| dataset                                                                                                                          | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| -------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- | --- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_dev.fr_decode_pytorch_transformer_pretrain  | **17.45** | 49.5   | 23.1   | 12.6   | 7.1    | 0.976 | 0.976 | 21880   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_test.fr_decode_pytorch_transformer_pretrain | **16.22** | 48.4   | 22.0   | 11.6   | 6.2    | 0.974 | 0.975 | 42798   | 43904   | s   |

- NOTE: shorten the total number epochs when pre-training the model: 100ep->30ep

### train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_asrtrans (3)

| dataset                                                                                                                  | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------------------------------------------------------------ | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_dev.fr_decode_pytorch_transformer_pretrain  | **16.50** | 47.5   | 21.5   | 11.5   | 6.3    | 1.000 | 1.012 | 22669   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_test.fr_decode_pytorch_transformer_pretrain | **15.53** | 47.0   | 20.9   | 10.8   | 5.6    | 0.994 | 0.995 | 43663   | 43904   |

- NOTE: shorten the total number epochs when pre-training the model: 100ep->30ep

### train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000

| dataset                                                                                                                | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ---------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_dev.fr_decode_pytorch_transformer  | **16.66** | 47.9   | 21.9   | 11.7   | 6.4    | 0.996 | 0.996 | 22308   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_test.fr_decode_pytorch_transformer | **15.47** | 47.0   | 20.9   | 10.8   | 5.8    | 0.981 | 0.981 | 43076   | 43904   |

### train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000

| dataset                                                                                                          | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ---------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_dev.fr_decode_pytorch_transformer  | **16.24** | 46.7   | 21.3   | 11.3   | 6.2    | 1.000 | 1.035 | 23190   | 22408   |
| exp/train_sp.fr_lc_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_test.fr_decode_pytorch_transformer | **15.30** | 46.1   | 20.4   | 10.5   | 5.5    | 1.000 | 1.012 | 44448   | 43904   |

# RNN results

### train_sp.fr_lc_pytorch_train_asrtrans_mttrans

| dataset                                                                 | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ----------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_asrtrans_mttrans/decode_dev.fr_decode  | **17.59** | 48.6   | 23.0   | 12.5   | 7.1    | 0.992 | 0.992 | 22273   | 22462   |
| exp/train_sp.fr_lc_pytorch_train_asrtrans_mttrans/decode_test.fr_decode | **16.68** | 48.1   | 22.0   | 11.9   | 6.6    | 0.984 | 0.984 | 43389   | 44080   |

### train_sp.fr_lc_pytorch_train_asrtrans

| dataset                                                         | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_asrtrans/decode_dev.fr_decode  | **17.79** | 48.4   | 22.8   | 12.6   | 7.3    | 0.995 | 0.995 | 22357   | 22462   |
| exp/train_sp.fr_lc_pytorch_train_asrtrans/decode_test.fr_decode | **16.30** | 47.3   | 21.6   | 11.4   | 6.2    | 0.993 | 0.993 | 43772   | 44080   |

### train_sp.fr_lc_pytorch_train

| dataset                                                         | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.fr_lc_pytorch_train_asrtrans/decode_dev.fr_decode  | **16.67** | 46.9   | 21.6   | 11.7   | 6.5    | 1.000 | 1.009 | 22668   | 22462   |
| exp/train_sp.fr_lc_pytorch_train_asrtrans/decode_test.fr_decode | **15.71** | 45.9   | 20.8   | 10.9   | 5.9    | 1.000 | 1.010 | 44533   | 44080   |
