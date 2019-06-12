# VGGBLSTMP(elayers=4, eunits=1024) + joint ctc decoding + lm rescoring
- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Tue Jun 11 11:22:28 JST 2019`
  - system information: `Linux chikaku1.sp.m.is.nagoya-u.ac.jp 3.10.0-862.14.4.el7.x86_64 #1 SMP Wed Sep 26 15:12:11 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.6.5`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.0`
  - Git hash: `2679f69a46f5cfa103325ceffb53793d6af4af2b`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
    - training config file: `./conf/train.yaml`
    - decoding config file: `./conf/decode.yaml`
    - cmvn file: `./data/train_trim_sp/cmvn.ark`
    - e2e file: `./exp/train_trim_sp_pytorch_train/results/model.acc.best`
    - e2e JSON file: `./exp/train_trim_sp_pytorch_train/results/model.json`
    
- Results

```
* speaker-adaptation datasets
# VGGBLSTMP CER (elayers=4, eunits=1024, BPE=500) + joint ctc decoding + lm rescoring
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.txt:|           SPKR                                 |            # Snt                       # Wrd            |           Corr                          Sub                         Del                         Ins                          Err                       S.Err            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.txt:|           Sum/Avg                              |            2710                       105941            |           83.1                          5.9                        11.0                         3.1                         20.0                        85.2            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.txt:|            SPKR                                |           # Snt                        # Wrd            |           Corr                          Sub                         Del                          Ins                         Err                        S.Err            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.txt:|            Sum/Avg                             |           2582                        100177            |           83.7                          5.1                        11.3                          2.6                        18.9                         83.5            |

# VGGBLSTMP WER(elayers=4, eunits=1024, BPE=500) + joint ctc decoding + lm rescoring
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            SPKR                                 |            # Snt                       # Wrd            |            Corr                         Sub                          Del                          Ins                          Err                        S.Err            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            Sum/Avg                              |            2710                        50251            |            85.7                         8.3                          6.0                          1.8                         16.1                         85.2            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            SPKR                                |            # Snt                       # Wrd            |            Corr                          Sub                           Del                          Ins                          Err                        S.Err            |
exp/train_trim_sp_pytorch_vggblstmp_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs35_mli600_mlo150/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm1.0_2layer_unit650_sgd_bs512/result.wrd.txt:|            Sum/Avg                             |            2582                        45445            |            86.5                          7.3                           6.1                          1.7                         15.1                         83.4            |


* legacy datasets
write a CER (or TER) result in exp/train_trim_sp_pytorch_train/decode_dev_decode/result.txt
| SPKR                      | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                   |  507   35090 | 92.3    5.1    2.5    8.8   16.5   91.3 |
write a WER result in exp/train_trim_sp_pytorch_train/decode_dev_decode/result.wrd.txt
| SPKR                      | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                   |  507  17783 | 90.3    7.3    2.4    4.5   14.3   91.3 |

write a CER (or TER) result in exp/train_trim_sp_pytorch_train/decode_test_decode/result.txt
| SPKR                  | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1155  52311 | 92.3    5.0    2.6   12.6   20.3   91.8 |
write a WER result in exp/train_trim_sp_pytorch_train/decode_test_decode/result.wrd.txt
| SPKR                  | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1155  27500 | 90.1    7.4    2.4    5.1   15.0   91.8 |
```
