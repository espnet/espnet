# pytorch large Transformer with specaug (4 GPUs) + Transformer LM (4 GPUs)

- Environments
  - date: `Mon July  27 2020`
  - python version: `3.7.3  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.0`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `To be added`
  - Commit date: `Aug 5, 2020`

- Model files 
    - model link: `To be added`
    - training config file: `./conf/train.yaml`
    - decoding config file: `./conf/decode.yaml`
    - cmvn file: `./data/train_960/cmvn.ark`
    - e2e file: `./librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best`
    - e2e JSON file: `./librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.json`
    - lm file: `./exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4/rnnlm.model.best`
    - lm JSON file: `./exp/train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4/model.json`
    - dict file: `./data/lang_char`
- Results 

best val scores = [0.90567905 0.9056724  0.90525274 0.90449862 0.90426245]
selected epochs = [36 54 41 22 20]
average over ['exp/train_trim_sp_pytorch_2_specaug/results/snapshot.ep.36', 'exp/train_trim_sp_pytorch_2_specaug/results/snapshot.ep.54', 'exp/train_trim_sp_pytorch_2_specaug/results/snapshot.ep.41', 'exp/train_trim_sp_pytorch_2_specaug/results/snapshot.ep.22', 'exp/train_trim_sp_pytorch_2_specaug/results/snapshot.ep.20']

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
