# Conformer (large model + specaug) (4 GPUs)
- KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition
    - Database: https://aihub.or.kr/aidata/105
    - Paper: https://www.mdpi.com/2076-3417/10/19/6936
    - This corpus contains 969 h of general open-domain dialog utterances, spoken by 2000 native Korean speakers.

## Environments
- date: `Tue Nov  3 00:22:13 EST 2020`
- python version: `3.8.3 (default, May 19 2020, 18:47:26)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.4`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.4.0`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: ([pretrained model](https://drive.google.com/file/d/1A2Gg18v-_z3dcw794aKL7SrPKHTyDbOU/view?usp=sharing))
    - training config file: `conf/tuning/train_pytorch_conformer_large.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_pytorch_conformer_large_specaug/results/model.val5.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_pytorch_conformer_large_specaug/results/model.json`
    - dict file: `data/lang_char`

### CER
```
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.txt
|   SPKR                 |   # Snt     # Wrd   |   Corr       Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg              |   3000      68439   |   94.4       3.4        2.2        1.8        7.4       58.8   |
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.txt
|   SPKR                 |   # Snt     # Wrd   |   Corr       Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg              |   3000      95592   |   93.8       3.8        2.4        2.2        8.5       72.3   |
```
### WER
```
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.wrd.txt
|   SPKR                 |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err       S.Err   |
|   Sum/Avg              |   3000       20401   |   83.0       14.1         2.9        3.8       20.8        58.8   |
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.wrd.txt
|   SPKR                 |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err       S.Err   |
|   Sum/Avg              |   3000       26621   |   79.6       17.4         3.0        5.1       25.5        72.3   |
```
### sWER (space-normalized WER)
- This metric was measured from space-normalized texts, which was performed only on the hypothesis text, based on spaces in the reference text. A more detailed description is given in our paper.
```
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.wrd.sp_norm.txt
|   SPKR                  |   # Snt       # Wrd    |   Corr         Sub         Del          Ins         Err       S.Err    |
|   Sum/Avg               |   3000        20401    |   88.2        10.5         1.4          1.4        13.2        50.2    |
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.wrd.sp_norm.txt
|   SPKR                  |   # Snt       # Wrd    |   Corr         Sub         Del          Ins         Err       S.Err    |
|   Sum/Avg               |   3000        26621    |   86.1        12.5         1.4          1.5        15.4        62.0    |
```

# Transformer (large model + specaug) (4 GPUs)

## Environments
- date: `Tue Nov  3 00:22:13 EST 2020`
- python version: `3.8.3 (default, May 19 2020, 18:47:26)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.4`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.4.0`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: ([pretrained model](https://drive.google.com/file/d/1BpEXi90SZxiX52Ent2P_lgFz5rwh1ryG/view?usp=sharing))
    - training config file: `conf/tuning/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/results/model.val5.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/results/model.json`
    - dict file: `data/lang_char`

### CER
```
exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.txt
|   SPKR                  |   # Snt      # Wrd   |   Corr         Sub         Del         Ins        Err       S.Err    |
|   Sum/Avg               |   3000       68439   |   94.1         3.5         2.4         1.9        7.8        59.7    |
exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.txt
|   SPKR                  |   # Snt      # Wrd   |   Corr         Sub         Del         Ins        Err       S.Err    |
|   Sum/Avg               |   3000       95592   |   93.6         3.9         2.5         2.2        8.5        72.0    |
```
### WER
```
exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.wrd.txt
|   SPKR                  |   # Snt       # Wrd    |   Corr         Sub         Del          Ins         Err       S.Err    |
|   Sum/Avg               |   3000        20401    |   82.4        14.4         3.1          3.9        21.4        59.7    |
exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.wrd.txt
|   SPKR                  |   # Snt       # Wrd    |   Corr         Sub         Del          Ins         Err       S.Err    |
|   Sum/Avg               |   3000        26621    |   79.5        17.4         3.1          5.0        25.5        71.9    |
```
### sWER (space-normalized WER)
```
exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/decode_eval_clean_model.val5.avg.best_decode_lm/result.wrd.sp_norm.txt
|    SPKR                  |    # Snt       # Wrd    |    Corr          Sub           Del          Ins          Err        S.Err    |
|    Sum/Avg               |    3000        20401    |    87.7         10.7           1.6          1.4         13.8         51.2    |
exp/train_pytorch_train_pytorch_transformer_large_ngpu4_specaug/decode_eval_other_model.val5.avg.best_decode_lm/result.wrd.sp_norm.txt
|    SPKR                  |    # Snt       # Wrd    |    Corr          Sub           Del          Ins          Err        S.Err    |
|    Sum/Avg               |    3000        26621    |    86.0         12.6           1.4          1.5         15.5         61.9    |
```
