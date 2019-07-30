# pytorch large Transformer with specaug (4 GPUs) + Large LM

## Models
- Model files (archived to `train_960_pytorch_train_pytorch_transformer_large_ngpu4_specaug.tar.gz` by `$ pack_model.sh`)
- model link: https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6
- training config file: `conf/tuning/train_pytorch_transformer_large_ngpu4.yaml`
- decoding config file: `conf/tuning/decode_pytorch_transformer_large.yaml`
- cmvn file: `data/train_960/cmvn.ark`
- e2e file: `exp/train_960_pytorch_train_pytorch_transformer_large_ngpu4_specaug/results/model.val5.avg.best`
- e2e JSON file: `exp/train_960_pytorch_train_pytorch_transformer_large_ngpu4_specaug/results/model.json`
- lm file: `exp/irielm.ep11.last5.avg/rnnlm.model.best`
- lm JSON file: `exp/irielm.ep11.last5.avg/model.json`

## Environments
- date: `Thu Jul 18 16:15:33 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.4.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `f9f40861423ba9a9c9f5a45bd4369dbdb9b3bbf9`
  - Commit date: `Thu Jul 18 15:40:51 2019 +0900`

## WER

```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_model.val5.avg.best_decode_pytorch_transformer_large_lm_large|2703|54402|98.0|1.8|0.2|0.2|2.2|27.9|
|decode_dev_other_model.val5.avg.best_decode_pytorch_transformer_large_lm_large|2864|50948|95.1|4.3|0.6|0.6|5.6|44.9|
|decode_test_clean_model.val5.avg.best_decode_pytorch_transformer_large_lm_large|2620|52576|97.7|2.0|0.3|0.3|2.6|29.9|
|decode_test_other_model.val5.avg.best_decode_pytorch_transformer_large_lm_large|2939|52343|95.0|4.4|0.6|0.6|5.7|47.7|
```

# pytorch Transformer (accum grad 8, single GPU)
  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Wed Jun 19 16:58:42 EDT 2019`
    - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `b32af59f229b54801a2cf7e4b8a48cadccd5fe5a`
  - Model files (archived to model.v1.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1bOaOEIZBveERti0x6mnBYiNsn6MSRd2E
    - training config file: `conf/tuning/train_pytorch_transformer_lr5.0_ag8.v2.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
    - cmvn file: `data/train_960/cmvn.ark`
    - e2e file: `exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_unigram5000/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_unigram5000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/decode_dev_clean_decode_pytorch_transformer_lm/result.wrd.txt
|    SPKR           |   # Snt       # Wrd    |   Corr          Sub         Del          Ins         Err        S.Err    |
|    Sum/Avg        |   2703        54402    |   96.7          2.9         0.3          0.4         3.7         38.5    |
exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/decode_dev_other_decode_pytorch_transformer_lm/result.wrd.txt
|    SPKR           |   # Snt       # Wrd    |   Corr          Sub         Del          Ins         Err        S.Err    |
|    Sum/Avg        |   2864        50948    |   91.4          7.7         0.9          1.3         9.8         59.7    |
exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/decode_test_clean_decode_pytorch_transformer_lm/result.wrd.txt
|    SPKR           |   # Snt       # Wrd    |    Corr         Sub          Del          Ins         Err        S.Err    |
|    Sum/Avg        |   2620        52576    |    96.5         3.1          0.4          0.5         4.0         38.3    |
exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/decode_test_other_decode_pytorch_transformer_lm/result.wrd.txt
|    SPKR           |   # Snt       # Wrd    |    Corr         Sub          Del          Ins         Err        S.Err    |
|    Sum/Avg        |   2939        52343    |    91.3         7.8          0.9          1.3        10.0         62.8    |
```

# pytorch Transformer without any hyper-parameter tuning
## train_960_pytorch_transformer_conv2d_e12_unit2048_d6_unit2048_aheads4_dim256_mtlalpha0.3_noam_sampprob0.0_ngpu3_bs32_lr10.0_warmup25000_mli512_mlo150_epochs100_accum2_lennormFalse_lsmunigram0.1/
```
decode_dev_clean_beam20_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024/result.wrd.txt: 3.8
decode_dev_other_beam20_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024/result.wrd.txt: 9.9
decode_test_clean_beam20_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024/result.wrd.txt: 4.2
decode_test_other_beam20_emodel.last10.avg.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024/result.wrd.txt: 9.8
```

# pytorch VGG-3BLSTM 1024 units, #BPE 5000, latest RNNLM training with tuned decoding (ctc_weight=0.5, lm_weight=0.7), dropout 0.2
## train_960_pytorch_vggblstm_e5_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_drop0.2_adadelta_sampprob0.0_bs20_mli800_mlo150
## WER
```
decode_dev_clean_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 4.0
decode_dev_other_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 12.3
decode_test_clean_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 4.0
decode_test_other_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 12.7
```

# pytorch VGG-3BLSTM 1024 units, #BPE 5000, latest RNNLM training with tuned decoding (ctc_weight=0.5, lm_weight=0.7)
## train_960_pytorch_vggblstm_e5_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs20_mli800_mlo150
## WER
```
decode_dev_clean_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 4.2
decode_dev_other_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 12.5
decode_test_clean_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 4.2
decode_test_other_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.7_1layer_unit1024_sgd_bs1024: 13.6
```

# pytorch VGG-3BLSTM 1024 units, #BPE 5000 more layers with tuned decoding (ctc_weight=0.5, lm_weight=0.5)
## train_960_vggblstm_e5_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs24_mli800_mlo150_unigram5000
## WER
```
decode_dev_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.5: 4.5
decode_dev_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.5: 13.0
decode_test_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.5: 4.6
decode_test_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.5_rnnlm0.5: 13.7
```

# pytorch VGG-3BLSTM 1024 units, #BPE 2000 (motivated by the RWTH setup, thanks to Albert Zeyer, Rohit Prabhavalkar, and Kazuki Irie for their comments)
## train_960_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150_unigram2000
## WER
```
decode_dev_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3: 5.0
decode_dev_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3: 14.3
decode_test_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3: 5.0
decode_test_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3: 14.9
```

# pytorch, BLSTMP 8layers
## CER (numbers in parenthesis are ER for different lm_weight)
```
decode_dev_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| 2.9 (2.7 w/ 0.2, 2.7 w/ 0.3, 2.7 w/ 0.4)
decode_dev_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| 9.6 (9.2 w/ 0.2, 9.1 w/ 0.3, 9.0 w/ 0.4)
decode_test_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| 2.7 (2.6 w/ 0.2, 2.6 w/ 0.3, 2.6 w/ 0.4)
decode_test_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| 9.9 (9.6 w/ 0.2, 9.4 w/ 0.3, 9.3 w/ 0.4)
```
## WER
```
decode_dev_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.wrd.txt:| 7.7 (7.2 w/ 0.2, 7.1 w/ 0.3, 7.2 w/ 0.4)
decode_dev_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.wrd.txt:| 21.1 (19.6 w/ 0.2, 19.2 w/ 0.3, 18.9 w/ 0.4)
decode_test_clean_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.wrd.txt:| 7.7 (7.2 w/ 0.2, 7.2 w/ 0.3, 7.1 w/ 0.4)
decode_test_other_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.wrd.txt:| 21.9 (20.5 w/ 0.2, 20.0 w/ 0.3, 19.7 w/ 0.4)
```
