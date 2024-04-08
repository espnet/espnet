# Transformer Result
## Pytorch backend Transformer without any hyperparameter tuning
This code can be used for running experiments in [End-To-End Multi-Speaker Speech Recognition With Transformer](https://ieeexplore.ieee.org/abstract/document/9054029) paper.
 - Environments
    - date: `Tue May  5 16:56:33 EDT 2020`
    - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
    - espnet version: `espnet 0.6.0`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `ae73f9b88cbec70bfa1325dd77042bbc289f419e`
    - Commit date: `Sun May 3 20:15:00 2020 -0400`
 - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - training config file: `conf/train_multispkr_transformer.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/tr/cmvn.ark`
    - e2e file: `exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/results/model.last10.avg.best`
    - e2e JSON file: `exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_word65000/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_word65000/model.json`
 - Results
    - WSJ\_2mix (model link: https://drive.google.com/open?id=10DH\_HhgFyzamq3QXqayxHJZMtxEf2oyR)

   ```
   exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_cv_decode_lm_word65000_model.last10.avg.best/min_perm_result.json
   |  # Snt |   Corr     Sub     Del   Ins    Err   |
   |  503   |   13996    1989    441   317    16.72 |
   exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_tt_decode_lm_word65000_model.last10.avg.best/min_perm_result.wrd.json
   |  # Snt |   Corr     Sub     Del   Ins    Err   |
   |  333   |   10094    1055    169   200    12.58 |
   ```

    - WSJ0\_2mix (model link: https://drive.google.com/open?id=1xm2W1AXBgnccq-AFkp5-sDe2R0X8CMNx)

   ```
   exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_cv_decode_lm_word65000_model.last10.avg.best/min_perm_result.json
   |  # Snt |   Corr     Sub     Del   Ins   Err |
   |  503   |   89863    3417    3568  1704  8.97 |
   exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_tt_decode_lm_word65000_model.last10.avg.best/min_perm_result.wrd.json
   |  # Snt |   Corr     Sub     Del   Ins    Err   |
   |  3000  |   83533    12552   2528  2394   17.72 |
   ```


# RNN Result
## WSJ-2mix
### CER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 85377 | 6581 | 4890 | 3982 | 15.96 |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 60849 | 3537 | 2695 | 1920 | 12.15 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 85388 | 5927 | 5533 | 2875 | 14.80 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 61630 | 3176 | 2275 | 1842 | 10.87 |
### WER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 12691 | 3169 | 566 | 651 | 26.70 |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 9350 | 1677 | 291 | 308 | 20.11 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 12817 | 2921 | 688 | 475 | 24.86 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 9557 | 1501 | 260 | 308 | 18.28 |

The mixture scheme is in the local/wsj_mix_scheme.tar.gz
Click to get the [pretrained model without speaker parallel attention](https://drive.google.com/open?id=11SWTPG5ggMHtqucHDTeWpNCRXrYMw4SZ).
Click to get the [pretrained model with speaker parallel attention]().

## WSJ0-2mix
### CER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 3000 | 531894 | 36189 | 30213 | 18161 | 14.13 |
### WER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 3000 | 79432 | 15538 | 3643 | 2744 | 22.23 |

Click to get the [pretrained model without speaker parallel attention](https://drive.google.com/open?id=1yiinAMHczS3JpK5b5bnt-BKqH1AMTFjH).
