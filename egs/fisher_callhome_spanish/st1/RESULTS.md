# NOTE: apostrophe is included both in hyp and ref

# Summary (4-gram BLEU)

| model                                                         | fisher_dev | fisher_dev2 | fisher_test | callhome_devtest | callhome_evltest |
| ------------------------------------------------------------- | ---------- | ----------- | ----------- | ---------------- | ---------------- |
| RNN (char) [[Weiss et al.]](https://arxiv.org/abs/1703.08581) | 48.30      | 49.10       | 48.70       | 16.80            | 17.40            |
| RNN (char)                                                    | 40.42      | 41.49       | 41.51       | 14.10            | 14.20            |
| RNN (BPE1k)                                                   | 30.96      | 31.56       | 31.31       | 9.74             | 10.30            |
| RNN (BPE1k) + ASR-MTL                                         | 36.54      | 36.99       | 35.57       | 12.19            | 12.66            |
| Transformer (char) + ASR-MTL                                  | 45.51      | 46.64       | 45.61       | 17.10            | 16.60            |
| Transformer (BPE1k) + ASR-MTL                                 | 46.64      | 47.64       | 46.45       | 16.80            | 16.80            |
| Transformer (BPE1k) + ASR-MTL + MT-MTL                        | 47.17      | 48.20       | 46.99       | 17.51            | 17.64            |
| Transformer (BPE1k) + ASR-PT                                  | 46.25      | 47.11       | 46.21       | 17.35            | 16.94            |
| Transformer (BPE1k) + ASR-PT + MT-PT                          | 46.25      | 47.60       | 46.72       | 17.62            | 17.50            |
| Transformer (BPE1k) + ASR-PT + MT-PT + SpecAugment            | 48.94      | 49.32       | 48.39       | 18.83            | 18.67            |
| Conformer (BPE1k) + ASR-PT + MT-PT + SpecAugment              | **51.14**  | **51.59**   | **51.03**   | **19.97**        | **20.44**        |

# Conformer results

### train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans

| dataset                                                                                                                                      | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| -------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/decode_fisher_dev.en_decode_pytorch_transformer       | **51.14** | 79.4   | 59.7   | 44.4   | 32.5   | 1.000 | 1.010 | 39996   | 39600   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/decode_fisher_dev2.en_decode_pytorch_transformer      | **51.59** | 79.8   | 60.2   | 44.8   | 32.9   | 1.000 | 1.010 | 39498   | 39101   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/decode_fisher_test.en_decode_pytorch_transformer      | **51.03** | 80.6   | 60.0   | 44.1   | 31.8   | 1.000 | 1.015 | 39397   | 38825   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/decode_callhome_devtest.en_decode_pytorch_transformer | **19.97** | 49.2   | 25.7   | 14.7   | 8.6    | 1.000 | 1.003 | 37524   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/decode_callhome_evltest.en_decode_pytorch_transformer | **20.44** | 49.3   | 26.3   | 15.2   | 9.2    | 0.991 | 0.991 | 18299   | 18457   |

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans.tar.gz by `$ pack_model.sh`)
  - training config file: `conf/tuning/train_pytorch_conformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_pretrain.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_conformer_bpe1000_specaug_asrtrans_mttrans/results/model.json`
  - NOTE: This model is initialized with the Transformer ASR model (BPE1k, use SpecAugment) on the encoder side and Transformer MT model (BPE1k) on the decoder side.
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

# Transformer results

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans

| dataset                                                                                                                                        | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_fisher_dev.en_decode_pytorch_transformer       | **48.94** | 77.5   | 57.5   | 42.2   | 30.5   | 1.000 | 1.013 | 40630   | 40118   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_fisher_dev2.en_decode_pytorch_transformer      | **49.32** | 77.7   | 57.7   | 42.6   | 31.0   | 1.000 | 1.019 | 40229   | 39482   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_fisher_test.en_decode_pytorch_transformer      | **48.39** | 78.3   | 57.2   | 41.4   | 29.6   | 1.000 | 1.025 | 40334   | 39357   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_callhome_devtest.en_decode_pytorch_transformer | **18.83** | 47.1   | 24.3   | 13.8   | 7.9    | 1.000 | 1.016 | 38018   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/decode_callhome_evltest.en_decode_pytorch_transformer | **18.67** | 46.5   | 23.9   | 13.6   | 8.0    | 1.000 | 1.014 | 18716   | 18457   |

- Model files (archived to train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1hawp5ZLw4_SIHIT3edglxbKIIkPVe8n3
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer_pretrain.yaml`
  - preprocess config file: `conf/specaug.yaml`
  - cmvn file: `data/train_sp.en/cmvn.ark`
  - e2e file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/results/model.val5.avg.best`
  - e2e JSON file: `exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_specaug_asrtrans_mttrans/results/model.json`
  - NOTE: This model is initialized with the Transformer ASR model (BPE1k, use SpecAugment) on the encoder side and Transformer MT model (BPE1k) on the decoder side.
- Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans

| dataset                                                                                                                                | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| -------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_fisher_dev.en_decode_pytorch_transformer       | **46.25** | 75.9   | 54.7   | 39.4   | 28.0   | 1.000 | 1.016 | 40746   | 40107   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_fisher_dev2.en_decode_pytorch_transformer      | **47.60** | 77.0   | 56.1   | 40.8   | 29.2   | 1.000 | 1.014 | 40042   | 39497   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_fisher_test.en_decode_pytorch_transformer      | **46.72** | 77.3   | 55.5   | 39.6   | 28.0   | 1.000 | 1.021 | 40222   | 39383   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_callhome_devtest.en_decode_pytorch_transformer | **17.62** | 45.9   | 23.0   | 12.7   | 7.2    | 0.999 | 0.999 | 37391   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans_mttrans/decode_callhome_evltest.en_decode_pytorch_transformer | **17.50** | 45.8   | 22.7   | 12.6   | 7.3    | 0.996 | 0.996 | 18375   | 18457   |

- NOTE: shorten the total number epochs when pre-training the model: 30ep->20ep

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans

| dataset                                                                                                                        | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------------------------------------------------------------------ | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_fisher_dev.en_decode_pytorch_transformer       | **46.25** | 76.2   | 55.0   | 39.3   | 27.8   | 1.000 | 1.008 | 40277   | 39962   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_fisher_dev2.en_decode_pytorch_transformer      | **47.11** | 76.7   | 55.8   | 40.3   | 28.6   | 1.000 | 1.014 | 39856   | 39287   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_fisher_test.en_decode_pytorch_transformer      | **46.21** | 77.3   | 55.1   | 39.1   | 27.4   | 1.000 | 1.021 | 40073   | 39248   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_callhome_devtest.en_decode_pytorch_transformer | **17.35** | 46.1   | 22.8   | 12.5   | 7.0    | 0.996 | 0.996 | 37281   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_bpe1000_asrtrans/decode_callhome_evltest.en_decode_pytorch_transformer | **16.94** | 45.4   | 22.4   | 12.2   | 6.9    | 0.990 | 0.990 | 18268   | 18457   |

- NOTE: shorten the total number epochs when pre-training the model: 30ep->20ep

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000

| dataset                                                                                                                               | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_fisher_dev.en_decode_pytorch_transformer       | **47.17** | 77.4   | 56.1   | 40.2   | 28.4   | 1.000 | 1.002 | 39710   | 39647   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_fisher_dev2.en_decode_pytorch_transformer      | **48.20** | 77.8   | 56.9   | 41.4   | 29.5   | 1.000 | 1.009 | 39380   | 39037   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_fisher_test.en_decode_pytorch_transformer      | **46.99** | 78.2   | 56.0   | 39.9   | 27.9   | 1.000 | 1.013 | 39290   | 38803   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_callhome_devtest.en_decode_pytorch_transformer | **17.51** | 46.7   | 23.3   | 12.8   | 7.2    | 0.984 | 0.984 | 36809   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.2_mt0.2_bpe1000/decode_callhome_evltest.en_decode_pytorch_transformer | **17.64** | 46.5   | 23.2   | 12.9   | 7.6    | 0.979 | 0.979 | 18069   | 18457   |

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000

| dataset                                                                                                                         | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_fisher_dev.en_decode_pytorch_transformer       | **46.64** | 76.8   | 55.4   | 39.8   | 28.1   | 0.999 | 0.999 | 39616   | 39669   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_fisher_dev2.en_decode_pytorch_transformer      | **47.64** | 77.4   | 56.4   | 40.7   | 29.0   | 1.000 | 1.007 | 39193   | 38933   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_fisher_test.en_decode_pytorch_transformer      | **46.45** | 77.7   | 55.4   | 39.2   | 27.5   | 1.000 | 1.010 | 39135   | 38741   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_callhome_devtest.en_decode_pytorch_transformer | **16.80** | 46.0   | 22.6   | 12.1   | 6.9    | 0.979 | 0.980 | 36651   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_ctcasr0.3_bpe1000/decode_callhome_evltest.en_decode_pytorch_transformer | **16.80** | 45.8   | 22.2   | 12.4   | 7.1    | 0.970 | 0.970 | 17904   | 18457   |

### train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctcasr0.3_bpe53

| dataset                                                                                                                                 | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctcasr0.3_bpe53/decode_fisher_dev.en_decode_pytorch_transformer_char       | **45.51** | 75.8   | 54.3   | 38.6   | 27.1   | 1.000 | 1.016 | 40943   | 40279   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctcasr0.3_bpe53/decode_fisher_dev2.en_decode_pytorch_transformer_char      | **46.64** | 76.6   | 55.3   | 39.6   | 28.2   | 1.000 | 1.018 | 40233   | 39508   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctcasr0.3_bpe53/decode_fisher_test.en_decode_pytorch_transformer_char      | **45.61** | 77.0   | 54.7   | 38.4   | 26.7   | 1.000 | 1.026 | 40451   | 39441   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctcasr0.3_bpe53/decode_callhome_devtest.en_decode_pytorch_transformer_char | **17.10** | 45.8   | 22.6   | 12.3   | 6.7    | 1.000 | 1.008 | 37717   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_pytorch_transformer_char_ctcasr0.3_bpe53/decode_callhome_evltest.en_decode_pytorch_transformer_char | **16.60** | 45.3   | 22.0   | 11.7   | 6.5    | 1.000 | 1.005 | 18557   | 18457   |

# RNN results

### train_sp.en_lc.rm_pytorch_train_rnn_ctcasr0.3_bpe1000

| dataset                                                                                         | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ----------------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_ctcasr0.3_bpe1000/decode_fisher_dev.en_decode_rnn       | **36.54** | 68.5   | 44.9   | 29.7   | 19.5   | 1.000 | 1.032 | 41512   | 40226   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_ctcasr0.3_bpe1000/decode_fisher_dev2.en_decode_rnn      | **36.99** | 68.6   | 45.3   | 30.2   | 19.9   | 1.000 | 1.042 | 41243   | 39593   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_ctcasr0.3_bpe1000/decode_fisher_test.en_decode_rnn      | **35.57** | 68.8   | 44.1   | 28.6   | 18.4   | 1.000 | 1.050 | 41540   | 39565   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_ctcasr0.3_bpe1000/decode_callhome_devtest.en_decode_rnn | **12.19** | 39.3   | 16.8   | 8.2    | 4.1    | 1.000 | 1.017 | 38052   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_ctcasr0.3_bpe1000/decode_callhome_evltest.en_decode_rnn | **12.66** | 39.0   | 17.1   | 8.5    | 4.5    | 1.000 | 1.005 | 18557   | 18457   |

### train_sp.en_lc.rm_pytorch_train_rnn_bpe1000

| dataset                                                                               | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------------------------- | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_fisher_dev.en_decode_rnn       | **30.96** | 63.8   | 39.0   | 24.3   | 15.2   | 1.000 | 1.034 | 41550   | 40188   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_fisher_dev2.en_decode_rnn      | **31.56** | 64.2   | 39.6   | 25.1   | 15.5   | 1.000 | 1.044 | 41442   | 39711   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_fisher_test.en_decode_rnn      | **31.31** | 65.1   | 39.4   | 24.6   | 15.2   | 1.000 | 1.045 | 41381   | 39614   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_callhome_devtest.en_decode_rnn | **9.74**  | 35.3   | 13.8   | 6.3    | 3.0    | 1.000 | 1.017 | 38063   | 37416   |
| exp/train_sp.en_lc.rm_pytorch_train_rnn_bpe1000/decode_callhome_evltest.en_decode_rnn | **10.30** | 35.2   | 14.2   | 6.7    | 3.4    | 1.000 | 1.018 | 18788   | 18457   |

### train_sp.en_lc_pytorch_train

| dataset                                                            | BLEU      | 1-gram | 2-gram | 3-gram | 4-gram | BP    | ratio | hyp_len | ref_len |
| ------------------------------------------------------------------ | --------- | ------ | ------ | ------ | ------ | ----- | ----- | ------- | ------- |
| exp/train_sp.en_lc_pytorch_train/decode_fisher_dev.en_decode       | **40.42** | 71.4   | 49.0   | 33.6   | 22.7   | 1.000 | 1.018 | 40695   | 39981   |
| exp/train_sp.en_lc_pytorch_train/decode_fisher_dev2.en_decode      | **41.49** | 71.9   | 49.9   | 34.8   | 23.8   | 1.000 | 1.027 | 40285   | 39213   |
| exp/train_sp.en_lc_pytorch_train/decode_fisher_test.en_decode      | **41.51** | 72.9   | 50.1   | 34.6   | 23.5   | 1.000 | 1.034 | 40358   | 39049   |
| exp/train_sp.en_lc_pytorch_train/decode_callhome_devtest.en_decode | **14.10** | 41.3   | 19.0   | 9.8    | 5.2    | 0.996 | 0.996 | 37268   | 37424   |
| exp/train_sp.en_lc_pytorch_train/decode_callhome_evltest.en_decode | **14.20** | 41.4   | 19.1   | 10.0   | 5.5    | 0.982 | 0.982 | 18139   | 18463   |
