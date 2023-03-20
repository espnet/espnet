# pytorch large Transformer with specaug (4 GPUs) + Transformer LM (4 GPUs)

- Environments
  - date: `Mon Aug 17 08:31:35 +03 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.7.0`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `8c33d3774ecbdef47485dadacef5edc032ef5f50`
  - Commit date: `Tue Jun 2 20:58:08 2020 +0900`


- Model files
    - e2e model link: `To be added`
    - lm model link: `To be added`
    - lm config file: `./conf/lm_transformer.yaml`
    - training config file: `./conf/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `./conf/decode.yaml`
    - e2e JSON file: `./exp/train_trim_sp_pytorch_3_specaug/results/model.json`
    - lm JSON file: `./exp/exp/train_lm_pytorch_lm_transformer_unigram5000_segmented_text/model.json`
- Results

```
write a CER (or TER) result in exp/train_trim_sp_pytorch_2_specaug/decode_dev_model.val5.avg.best_decode_lm_transformer/result.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5002  111250 | 88.5    6.0    5.5    1.9   13.4   60.1 |
write a WER result in exp/train_trim_sp_pytorch_2_specaug/decode_dev_model.val5.avg.best_decode_lm_transformer/result.wrd.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5002   60166 | 86.5    8.6    5.0    1.1   14.6   59.3 |
write a CER (or TER) result in exp/train_trim_sp_pytorch_2_specaug/decode_test_model.val5.avg.best_decode_lm_transformer/result.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5365  119010 | 87.8    6.2    6.1    2.2   14.4   60.8 |
write a WER result in exp/train_trim_sp_pytorch_2_specaug/decode_test_model.val5.avg.best_decode_lm_transformer/result.wrd.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5365   64297 | 86.9    8.4    4.7    1.2   14.2   60.1 |

```

# pytorch large Transformer with specaug (4 GPUs) + RNN LM


- Model files
    - e2e model link: `To be added`
    - lm model link: `To be added`
    - lm config file: `./conf/lm.yaml`
    - training config file: `./conf/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `./conf/decode.yaml`
    - e2e JSON file: `./exp/train_trim_sp_pytorch_3_specaug/results/model.json`
    - lm JSON file: `./exp/exp/train_lm_unigram5000_segmented_text/model.json`
- Results

```
write a CER (or TER) result in exp/train_trim_sp_pytorch_2_specaug/decode_dev_model.val5.avg.best_decode_lm_transformer/result.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5002  111396 | 87.6    6.8    5.5    2.2   14.6   62.7 |
write a WER result in exp/train_trim_sp_pytorch_2_specaug/decode_dev_model.val5.avg.best_decode_lm_transformer/result.wrd.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5002   60166 | 85.5   9.6     4.8    1.2   15.6   61.7 |
write a CER (or TER) result in exp/train_trim_sp_pytorch_2_specaug/decode_test_model.val5.avg.best_decode_lm_transformer/result.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5365   119010| 87.2   6.6     6.2    2.2   15.1   63.4 |
write a WER result in exp/train_trim_sp_pytorch_2_specaug/decode_test_model.val5.avg.best_decode_lm_transformer/result.wrd.txt
| SPKR                                          | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                                       | 5365   64297 | 86.3   9.0     4.7    1.2   14.9   62.4 |

```
