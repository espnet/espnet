# Transformer (PyTorch 1.3 + builtin CTC)
- Environments
  - date: `Wed Jan 22 23:33:17 CST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.3.1`
  - Git hash: `28ce90a17148afcb36f4e593966911b9c3a6230b`
    - Commit date: `Tue Jan 7 16:34:06 2020 +0900`
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1Az-4H25uwnEFa4lENc-EKiPaWXaijcJp
  - training config file: `conf/tuning/train_pytorch_transformer.yaml`
  - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
  - cmvn file: `data/train_si284/cmvn.ark`
  - e2e file: `exp/train_si284_pytorch_train_no_preprocess/results/model.acc.best`
  - e2e JSON file: `exp/train_si284_pytorch_train_no_preprocess/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word65000/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word65000/model.json`
  - dict file: `data/lang_1char/train_si284_units.txt
- Results

  ### CER

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_dev93_decode_lm_word65000|503|48634|96.7|1.4|1.8|0.8|4.1|65.6|
  |decode_test_eval92_decode_lm_word65000|333|33341|97.9|0.9|1.2|0.6|2.7|52.9|

  ### WER

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_dev93_decode_lm_word65000|503|8234|92.6|6.3|1.1|1.4|8.8|60.0|
  |decode_test_eval92_decode_lm_word65000|333|5643|95.4|4.1|0.5|1.0|5.6|44.1|

# RNN (PyTorch 1.3 + builtin CTC)
- Environments
  - date: `Wed Jan 22 23:33:17 CST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.3.1`
  - Git hash: `28ce90a17148afcb36f4e593966911b9c3a6230b`
    - Commit date: `Tue Jan 7 16:34:06 2020 +0900`
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1zIK6vE_0Yfn9ezcLwNPk1cCXhrrw9Oje
  - training config file: `conf/tuning/train_rnn.yaml`
  - decoding config file: `conf/tuning/decode_rnn.yaml`
  - cmvn file: `data/train_si284/cmvn.ark`
  - e2e file: `exp/train_si284_pytorch_train_rnn_no_preprocess/results/model.acc.best`
  - e2e JSON file: `exp/train_si284_pytorch_train_rnn_no_preprocess/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word65000/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word65000/model.json`
  - dict file: `data/lang_1char/train_si284_units.txt`
- Results

  ### CER

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_dev93_decode_rnn_lm_word65000|503|48634|96.7|1.6|1.7|0.7|4.0|67.0|
  |decode_test_eval92_decode_rnn_lm_word65000|333|33341|98.2|0.9|0.9|0.5|2.4|51.4|

  ### WER

  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_test_dev93_decode_rnn_lm_word65000|503|8234|92.4|6.6|1.0|1.2|8.8|61.4|
  |decode_test_eval92_decode_rnn_lm_word65000|333|5643|95.7|3.9|0.4|0.9|5.3|42.9|


# Transformer result
Be careful with patience !!!

## e12 e6 d_units 2048 Avg Last 10 pytorch backend + CTC/LM joint decoding (4GPU best: bs16 accum1)
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu4_bs16_lr10_warmup25000_dropout0.1_attndropout0.0_mli512_mlo150_epochs100_accum1_lennormFalse_lsmunigram0.1/decode_test_dev93_beam30_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.txt |503|48634|97.2|1.2|1.7|0.5|3.4|65.8|
|exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu4_bs16_lr10_warmup25000_dropout0.1_attndropout0.0_mli512_mlo150_epochs100_accum1_lennormFalse_lsmunigram0.1/decode_test_dev93_beam30_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.txt |333|33341|98.4|0.7|0.9|0.4|1.9|49.5|

### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu4_bs16_lr10_warmup25000_dropout0.1_attndropout0.0_mli512_mlo150_epochs100_accum1_lennormFalse_lsmunigram0.1/decode_test_dev93_beam30_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.wrd.txt |503|8234|92.9|5.8|1.2|0.7|7.7|60.2|
|exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu4_bs16_lr10_warmup25000_dropout0.1_attndropout0.0_mli512_mlo150_epochs100_accum1_lennormFalse_lsmunigram0.1/decode_test_eval92_beam30_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.wrd.txt |333|5643|96.1|3.4|0.5|0.6|4.5|40.8|

## e12 e6 d_units 2048 Avg Last 10 pytorch backend + CTC/LM joint decoding (1GPU best: bs32, accum2)
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
| exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu1_bs32_lr10.0_warmup25000_mli512_mlo150_epochs100_accum2_lennormFalse_lsmunigram0.1/decode_test_dev93_beam10_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.txt |503|48634|97.0|1.2|1.8|0.7|3.7|63.8|
| exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu1_bs32_lr10.0_warmup25000_mli512_mlo150_epochs100_accum2_lennormFalse_lsmunigram0.1/decode_test_eval92_beam10_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.txt |333|33341|98.1|0.7|1.2|0.4|2.3|49.8|

### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu1_bs32_lr10.0_warmup25000_mli512_mlo150_epochs100_accum2_lennormFalse_lsmunigram0.1/decode_test_dev93_beam10_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.wrd.txt |503|8234|93.2|5.8|1.1|1.3|8.2|57.5|
|exp/train_si284_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu1_bs32_lr10.0_warmup25000_mli512_mlo150_epochs100_accum2_lennormFalse_lsmunigram0.1/decode_test_eval92_beam10_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.wrd.txt |333|5643|95.9|3.5|0.7|0.7|4.8|42.0|


## Big Model e12 e6 d_units 2048 Avg Last 10
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_dev93_beam1_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.0|503|48634|95.8|2.2|2.0|1.3|5.5|81.9|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_eval92_beam1_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.0|333|33341|97.2|1.7|1.2|1.1|4.0|76.6|

### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_dev93_beam1_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.0|503|8234|87.2|11.2|1.6|1.6|14.4|79.5|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_eval92_beam1_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.0|333|5643|89.9|9.1|1.0|1.2|11.4|72.4


## Big Model e12 e6 d_units 2048
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_dev93_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|503|48634|94.6|2.7|2.6|1.8|7.1|85.5|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_eval92_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|333|33341|96.1|2.1|1.8|1.4|5.3|78.7|

### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_dev93_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|503|8234|84.2|13.7|2.1|2.3|18.1|83.3|
|exp/train_si284_chainer_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.2/decode_test_eval92_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|333|5643|87.7|10.8|1.5|1.7|14.0|74.5|


## Base Model e6 d6 d_units 1024
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_dev93_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|503|48634|94.5|3.0|2.5|1.8|7.3|86.9|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_eval92_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|333|33341|96.0|2.2|1.8|1.7|5.7|83.8|

### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_dev93_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|503|8234|83.2|14.5|2.3|2.1|18.9|85.3|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_eval92_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0|333|5643|86.6|11.8|1.6|2.1|15.5|80.8|




## Transformer initial result

### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_dev93_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0_patience3|503 | 48634 | 92.6 | 4.0 | 3.4 | 2.9 | 10.3 | 91.7 |
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_eval92_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0_patience3|333|33341|94.4|3.2|2.5|2.2|7.9|90.1|

### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_dev93_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0_patience3|503|8234|77.8|19.2|3.0|3.7|25.9|90.1|
|exp/train_si284_chainer_transformer_conv2d_e6_unit1024_d6_unit1024_aheads4_dim256_mtlalpha0.0_noam_sampprob0.0_ngpu2_bs32_lr10.0_warmup70000_mli512_mlo150_epochs100_accum1_lennormTrue_lsmunigram0.1/decode_test_eval92_beam1_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.0_patience3|333|5643|82.0|16.2|1.9|3.2|21.2|88.0|


# RNN results
## (pytorch) 4-layer blstmp with no subsampling + wide rnn wordlm + add attention;
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_pytorch_blstmp_e4_subsample1_1_1_1_1_*_mtlalpha0.2_adadelta_bs15_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000|503|48634|97.4|1.2|1.3|0.6|3.2|59.8|
|exp/train_si284_pytorch_blstmp_e4_subsample1_1_1_1_1_*_mtlalpha0.2_adadelta_bs15_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000|333|33341|98.5|0.8|0.7|0.5|2.1|48.9|
### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_pytorch_blstmp_e4_subsample1_1_1_1_1_*_mtlalpha0.2_adadelta_bs15_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000|503|8234|94.0|5.1|0.9|1.0|7.0|53.1|
|exp/train_si284_pytorch_blstmp_e4_subsample1_1_1_1_1_*_mtlalpha0.2_adadelta_bs15_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_1layer_unit1000_sgd_bs300_word65000/result.wrd.txt:|333|5643|96.1|3.5|0.4|0.8|4.7|38.4|


## change hyperparameters mtlalpha: 0.5 -> 0.2 and beam_size: 20 -> 30
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.2_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam30_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|503|48634|96.7|1.5|1.8|0.7|4.0|65.0 |
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.2_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam30_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|333|33341|98.1|0.9|1.1|0.6|2.5|51.7|
### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.2_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam30_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|503|8234|92.5|6.5|1.0|1.2|8.7|59.0|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.2_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam30_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|333|5643|95.5|3.9|0.6|0.9|5.4|44.7|


## change hyperparameters mtlalpha: 0.5 -> 0.2 and beam_size: 20 -> 30
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|503|48634|96.1|2.1|1.8|1.2|5.1|74.0|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|333|33341|97.7|1.3|1.0|1.1|3.4|59.8|
### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|503|8234|90.1|8.8|1.0|1.9|11.7|70.8|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|333|5643|93.4|6.2|0.4|1.7|8.3|52.9|


## change character rnnlm to word rnnlm
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0 |503|48634|96.1|2.1|1.8|1.2|5.1|74.0|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|333|33341|97.7|1.3|1.0|1.1|3.4|59.8|
### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|503|8234|90.1|8.8|1.0|1.9|11.7|70.8|
|exp/train_si284_vggblstmp_e6_*_mtlalpha0.5_adadelta_bs30_mli800_mlo150_lsmunigram0.05/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_wordrnnlm1.0|333|5643|93.4|6.2|0.4|1.7|8.3|52.9|


## change character rnnlm to word rnnlm
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|503|48634|95.7|2.4|1.9|1.2|5.5|78.5|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|333|33341|97.2|1.4|1.3|1.0|3.8|67.0|
### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|503|8234|88.7|10.1|1.2|1.8|13.1|75.9|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|333|5643|92.2|7.1|0.7|1.5|9.3|60.4|


## combine rnnlm and joint attention/CTC decoding
### CER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|503|48634|95.7|2.4|1.9|1.2|5.5|78.5|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|333|33341|97.2|1.4|1.3|1.0|3.8|67.0|
### WER
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_dev93_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|503|8234|88.7|10.1|1.2|1.8|13.1|75.9|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_eval92_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0|333|5643|92.2|7.1|0.7|1.5|9.3|60.4|


## added rnnlm in a decoder network
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_dev93_beam20_eacc.best_p0.1_len0.3-0.0_rnnlm0.1|503|48634|94.2|3.3|2.5|2.5|8.3|88.9|
|exp/train_si284_a04_ch_vggblstmp_e6/decode_test_eval92_beam20_eacc.best_p0.1_len0.3-0.0_rnnlm0.1|333|33341|96.4|2.0|1.6|1.6|5.2|82.0|


## 4 -> 6 layers in the encoder network (now this is default)
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a03_ch_vggblstmp_e6/decode_test_dev93_beam20_eacc.best_p0_len0.3-0.8|503|48634|94.1|3.4|2.5|2.6|8.5|92.0|
|exp/train_si284_a03_ch_vggblstmp_e6/decode_test_eval92_beam20_eacc.best_p0_len0.3-0.8|333|33341|95.8|2.5|1.7|1.7|5.9|85.9|

## initial result. c.f. Watanabe et al IEEE JSTSP (2017) report 12.0% for dev93 and 8.3% for eval 92
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_si284_a09/decode_test_dev93_beam20_eacc.best_p0_len0.3-0.8|503 |49334| 92.3 |3.7 |4.0 |2.3 |10.1 |93.0|
|exp/train_si284_a09/decode_test_eval92_beam20_eacc.best_p0_len0.3-0.8 |333 |33740| 94.3 |2.7 |3.0 |1.8 |7.6 |90.1|
