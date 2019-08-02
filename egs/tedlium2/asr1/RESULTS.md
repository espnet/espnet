# Transformer (large model + specaug + large LM)

  - Model files (archived to large.tar.gz by `$ pack_model.sh`)
    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
    - training config file: `./conf/tuning/train_pytorch_transformer.v2_epochs100.yaml`
    - decoding config file: `./conf/decode_lm-weight0.5_beam-size40.yaml`
    - cmvn file: `./data/train_trim_sp/cmvn.ark`
    - e2e file: `./exp/train_trim_sp_pytorch_nbpe500_ngpu2_train_pytorch_transformer.v2_epochs100_specaug/results/model.last10.avg.best.ep86_irielmep1_beam40`
    - e2e JSON file: `./exp/train_trim_sp_pytorch_nbpe500_ngpu2_train_pytorch_transformer.v2_epochs100_specaug/results/model.json`
    - lm file: `./exp/train_rnnlm_pytorch_lm_irie_batchsize128_unigram500/rnnlm.model.best`
    - lm JSON file: `./exp/train_rnnlm_pytorch_lm_irie_batchsize128_unigram500/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_trim_sp_pytorch_nbpe500_ngpu2_train_pytorch_transformer.v2_epochs100_specaug/decode_dev_decode_lm-weight0.5_beam-size40.ep86_irielmep1_beam40/result.wrd.txt
|     SPKR                           |     # Snt          # Wrd      |     Corr              Sub             Del             Ins             Err           S.Err      |
|     Sum/Avg                        |      507           17783      |     91.9              4.9             3.2             1.2             9.3            73.6      |
exp/train_trim_sp_pytorch_nbpe500_ngpu2_train_pytorch_transformer.v2_epochs100_specaug/decode_test_decode_lm-weight0.5_beam-size40.ep86_irielmep1_beam40/result.wrd.txt
|      SPKR                       |     # Snt           # Wrd      |      Corr             Sub              Del              Ins             Err            S.Err      |
|      Sum/Avg                    |     1155            27500      |      92.8             3.8              3.3              0.9             8.1             65.8      |
```

## NOTE: contribution on WER

| system                     |   dev WER |   test WER |
| :------                    | --------: | ---------: |
| baseline (large model)     |      12.8 |       11.0 |
| w/ speed perturb (sp)      |      12.2 |       10.4 |
| w/ specaug                 |      11.2 |        9.6 |
| w/ specaug + sp            |      10.1 |        8.9 |
| w/ specaug + sp + large LM |       9.3 |        8.1 |

# Transformer (large model (12 elayers, 6 decoders, 2048 units))

  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Wed Jun  5 22:37:01 EDT 2019`
    - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `f1a69c2d6ffb34e4c008f951449157ef7caaf0e9`
  - Model files (archived to model.v2.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1rdRY-S7FbtkZuxwLMn2Z-Aps-q2bthEV
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_trim_sp/cmvn.ark`
    - e2e file: `../asr2/exp/train_trim_sp_pytorch_train_d6-2048_lr5.0/results/model.last10.avg.best`
    - e2e JSON file: `../asr2/exp/train_trim_sp_pytorch_train_d6-2048_lr5.0/results/model.json`
    - lm file: `../asr2/exp/train_rnnlm_pytorch_2layer_unit650_sgd_bs512_unigram500/rnnlm.model.best`
    - lm JSON file: `../asr2/exp/train_rnnlm_pytorch_2layer_unit650_sgd_bs512_unigram500/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
../asr2/exp/train_trim_sp_pytorch_train_d6-2048_lr5.0/decode_dev_decode_lmw0.3/result.wrd.txt
| SPKR                      | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                   |  507  17783 | 89.7    7.2    3.1    1.9   12.2   80.5 |
../asr2/exp/train_trim_sp_pytorch_train_d6-2048_lr5.0/decode_test_decode_lmw0.3/result.wrd.txt
| SPKR                  | # Snt  # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
| Sum/Avg               | 1155   27500 | 90.9     5.9    3.2     1.3   10.4    73.8 |
```

# Transformer (small decoder (6 layers 1024 units))

- Environments (obtained by `$ get_sys_info.sh`)
    - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `dde24df47c108fc4d3d52916da2a97032dce8153`
  - Model files (archived to model.v1.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=16GGLN0bvt0DdHBXqVaY4RySyYa32hqbd
    - training config file: `conf/train_d6.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_trim_sp/cmvn.ark`
    - e2e file: `exp/train_trim_sp_pytorch_train_d6/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_trim_sp_pytorch_train_d6/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_unigram500/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_unigram500/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_trim_sp_pytorch_train_d6/decode_dev_decode/result.wrd.txt
| SPKR                      | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                   |  507  17783 | 89.3    6.9    3.7    2.0   12.6   83.2 |
exp/train_trim_sp_pytorch_train_d6/decode_test_decode/result.wrd.txt
| SPKR                  | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1155  27500 | 90.0    5.9    4.1    1.4   11.5   76.9 |
```

# VGGBLSTMP (elayers=4, eunits=1024, BPE=500)  + joint ctc decoding + lm rescoring
```
$ grep -e Avg -e SPKR -m 2 exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/*/result.wrd.txt
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            SPKR                                 |            # Snt                       # Wrd            |            Corr                         Sub                          Del                          Ins                          Err                        S.Err            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            Sum/Avg                              |             507                        17783            |            89.1                         7.9                          3.0                          1.9                         12.8                         83.0            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            SPKR                             |            # Snt                        # Wrd            |            Corr                           Sub                          Del                           Ins                          Err                         S.Err            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            Sum/Avg                          |            1155                         27500            |            89.1                           7.4                          3.5                           1.7                         12.6                          75.7            |
```

# VGGBLSTMP (elayers=6) + joint ctc decoding + lm rescoring
```
exp/train_trim_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.0-0.0_ctcw0.3_rnnlm1.0/result.txt:|        Sum/Avg                          |         507                 95429        |        91.8                  4.2                  4.0                  2.7                 10.8                 89.3        |
exp/train_trim_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.0-0.0_ctcw0.3_rnnlm1.0/result.txt:|        Sum/Avg                       |        1155                145066         |        92.2                  3.7                   4.1                  2.4                  10.1                 85.3         |
exp/train_trim_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.0-0.0_ctcw0.3_rnnlm1.0/result.wrd.txt:|        Sum/Avg                           |         507                17783         |        83.2                 13.7                   3.1                  3.0                  19.8                 89.3         |
exp/train_trim_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.0-0.0_ctcw0.3_rnnlm1.0/result.wrd.txt:|        Sum/Avg                       |        1155                 27500         |        84.0                   12.3                   3.7                   2.6                  18.6                  85.3         |
```

# BLSTMP (elayers=6) CER
```
$ grep -e Avg -e SPKR -m 2 exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_*_beam20_eacc.best_p0.1_len0.1-0.8/result.txt
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       SPKR                            |       # Snt             # Wrd       |       Corr                Sub                Del               Ins                Err              S.Err       |
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       Sum/Avg                         |        507              95429       |       91.6                4.2                4.3               3.3               11.8               94.5       |
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       SPKR                        |       # Snt               # Wrd       |       Corr                Sub                 Del                Ins                Err               S.Err       |
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       Sum/Avg                     |       1155               145066       |       91.5                4.0                 4.4                3.0               11.5                92.0       |
```

# BLSTMP (elayers=6) WER
```
$ grep -e Avg -e SPKR -m 2 exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_*_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       SPKR                            |       # Snt              # Wrd       |       Corr                Sub                 Del                Ins                Err               S.Err       |
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       Sum/Avg                         |        507               17783       |       80.0               16.2                 3.8                4.0               23.9                94.5       |
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       SPKR                         |       # Snt              # Wrd        |       Corr                 Sub                Del                 Ins                 Err               S.Err        |
exp/train_trim_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       Sum/Avg                      |       1155               27500        |       80.4                15.5                4.1                 3.1                22.7                92.0        |
```

# VGGBLSTMP with chainer backend
```
$ grep -e Avg -e SPKR -m 2 exp/train_trim_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_*_beam20_eacc.best_p0.1_len0.1-0.8/result.txt
exp/train_trim_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       SPKR                            |       # Snt              # Wrd       |       Corr                Sub                 Del                Ins                Err              S.Err       |
exp/train_trim_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       Sum/Avg                         |        507               95429       |       91.4                4.2                 4.4                3.7               12.3               94.1       |
exp/train_trim_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       SPKR                         |       # Snt              # Wrd        |       Corr                 Sub                Del                 Ins                 Err              S.Err        |
exp/train_trim_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       Sum/Avg                      |       1155              145066        |       91.4                 4.1                4.4                 3.5                12.0               93.2        |
```

# BLSTMP (elayers=4) CER
```
grep -e Avg -e SPKR -m 2 exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_*_beam20_eacc.best_p0.1_len0.1-0.8/result.txt
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       SPKR                            |       # Snt             # Wrd       |       Corr                Sub                Del               Ins                Err              S.Err       |
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       Sum/Avg                         |        507              95429       |       90.9                4.7                4.5               3.5               12.6               94.1       |
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       SPKR                        |       # Snt               # Wrd       |       Corr                Sub                 Del                Ins                Err               S.Err       |
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.txt:|       Sum/Avg                     |       1155               145066       |       90.8                4.5                 4.7                3.2               12.4                93.1       |
```

# BLSTMP (elayers=4) WER
```
$ grep -e Avg -e SPKR -m 2 exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_*_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       SPKR                            |       # Snt              # Wrd       |       Corr                Sub                 Del                Ins                Err               S.Err       |
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       Sum/Avg                         |        507               17783       |       78.4               17.8                 3.8                4.0               25.6                94.1       |
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       SPKR                         |       # Snt              # Wrd        |       Corr                 Sub                Del                 Ins                 Err               S.Err        |
exp/train_trim_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_test_beam20_eacc.best_p0.1_len0.1-0.8/result.wrd.txt:|       Sum/Avg                      |       1155               27500        |       78.9                16.8                4.3                 3.3                24.4                93.1        |
```
