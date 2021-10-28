# conformer (only 20 epochs)
## Environments
- date: `Wed Aug  4 15:52:24 EDT 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.7.1`
- Git hash: `1fa7f0da2fcc8656feb5cb4325d562409ad23dbf`
  - Commit date: `Fri Jun 25 17:24:32 2021 -0400`

## train_nodup_sp_pytorch_train_pytorch_conformer_kernel15
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_lm|5413|56154|80.9|16.0|3.0|2.8|21.9|68.3|
|decode_train_dev_decode_lm|4000|47147|81.2|15.5|3.3|3.2|22.0|71.4|

# transformer (only 20 epochs)
  - This recipe seems to be over-trained at more than 20 epochs. It may require some tuning to avoid it.
  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Fri Jun 14 19:49:54 EDT 2019`
    - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `b32af59f229b54801a2cf7e4b8a48cadccd5fe5a`
  - Model files (archived to model.v2.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1x96BTGXpphdZU9_Ahb4hOqttQ5x-iuw1
    - training config file: `conf/tuning/train_pytorch_transformer.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
    - cmvn file: `data/train_nodup_sp/cmvn.ark`
    - e2e file: `exp/train_nodup_sp_pytorch_train_pytorch_transformer/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_nodup_sp_pytorch_train_pytorch_transformer/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_nodup_sp_pytorch_train_pytorch_transformer/decode_train_dev_decode_pytorch_transformer_lm/result.txt
|  SPKR                               | # Snt    # Wrd  |  Corr     Sub      Del      Ins     Err    S.Err  |
|  Sum/Avg                            | 4000     47147  |  79.1    17.1      3.8      3.2    24.1     73.6  |
exp/train_nodup_sp_pytorch_train_pytorch_transformer/decode_dev_decode_pytorch_transformer_lm/result.txt
| SPKR                               | # Snt   # Wrd  | Corr     Sub     Del     Ins     Err   S.Err  |
| Sum/Avg                            | 5413    56154  | 79.1    17.3     3.6     2.6    23.5    68.7  |
```

# transformer
  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Thu Jun 13 17:58:03 EDT 2019`
    - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `d299bf4c88d11dbce4aef8c28db2ffe7f48b7c07`
  - Model files (archived to model.v1.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1mcu-wPw7TOFnz0V__-_0jsTD_zDX_8QN
    - training config file: `conf/tuning/train_pytorch_transformer.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
    - cmvn file: `data/train_nodup_sp/cmvn.ark`
    - e2e file: `exp/train_nodup_sp_pytorch_train_pytorch_transformer/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_nodup_sp_pytorch_train_pytorch_transformer/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_nodup_sp_pytorch_train_pytorch_transformer/decode_train_dev_decode_pytorch_transformer_lm/result.txt
|  SPKR                               | # Snt    # Wrd  |  Corr     Sub      Del      Ins     Err    S.Err  |
|  Sum/Avg                            | 4000     47147  |  78.9    17.2      3.9      3.2    24.3     74.2  |
exp/train_nodup_sp_pytorch_train_pytorch_transformer/decode_dev_decode_pytorch_transformer_lm/result.txt
| SPKR                               | # Snt   # Wrd  | Corr     Sub     Del     Ins     Err   S.Err  |
| Sum/Avg                            | 5413    56154  | 78.9    17.5     3.6     2.7    23.8    69.1  |
```

# use wide and shallow network
```
$ rg -e Avg exp/train_nodup_sp_pytorch_vggblstm_e3_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_*0.6*0.3*/result.txt | sort | sed -E 's/ +/ /g'
exp/train_nodup_sp_pytorch_vggblstm_e3_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.6_rnnlm0.3/result.txt:| Sum/Avg | 5413 56154 | 75.3 20.5 4.3 2.7 27.4 72.2 |
exp/train_nodup_sp_pytorch_vggblstm_e3_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_train_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.6_rnnlm0.3/result.txt:| Sum/Avg | 3999 47130 | 74.2 20.7 5.0 3.0 28.8 76.8 |
```

# use RNNLM
```
$ grep Avg exp/train_nodup_sp_ch_vggblstmp_e8/decode_*_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.2/result.txt
exp/train_nodup_sp_ch_vggblstmp_e8/decode_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.2/result.txt:| Sum/Avg                            | 5413     56154  | 74.4    20.7     4.9      2.7    28.3    72.4  |
exp/train_nodup_sp_ch_vggblstmp_e8/decode_train_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.2/result.txt:|  Sum/Avg                            |  3999     47130  |  73.1    20.7      6.2      2.8     29.7     76.9  |
```

# use CTC/attention joint decoding
```
$grep -e Avg -e SPKR -m 2 exp/train_nodup_sp_ch_vggblstmp_e8/*/result.txt
exp/train_nodup_sp_ch_vggblstmp_e8/decode_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| SPKR                              | # Snt  # Wrd  | Corr    Sub    Del    Ins     Err  S.Err |
exp/train_nodup_sp_ch_vggblstmp_e8/decode_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| Sum/Avg                           | 5413   56154  | 74.1   21.9    4.0    3.0    28.9   73.7 |
exp/train_nodup_sp_ch_vggblstmp_e8/decode_train_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| SPKR                               | # Snt   # Wrd  | Corr     Sub    Del     Ins     Err   S.Err  |
exp/train_nodup_sp_ch_vggblstmp_e8/decode_train_dev_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3/result.txt:| Sum/Avg                            | 3999    47130  | 72.7    22.2    5.1     3.2    30.5    78.1  |
```

# change elayers 4 -> 8, penalty 0.0 -> 0.3
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_sp_a03_ch_vggblstmp_e8/*p0.3*/result.txt
exp/train_nodup_sp_a03_ch_vggblstmp_e8/decode_dev_beam20_eacc.best_p0.3_len0.0-0.8/result.txt:| SPKR                              | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/train_nodup_sp_a03_ch_vggblstmp_e8/decode_dev_beam20_eacc.best_p0.3_len0.0-0.8/result.txt:| Sum/Avg                           | 5413   56154 | 71.4   22.9    5.7    3.3   31.9   74.0 |
exp/train_nodup_sp_a03_ch_vggblstmp_e8/decode_train_dev_beam20_eacc.best_p0.3_len0.0-0.8/result.txt:| SPKR                              | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
exp/train_nodup_sp_a03_ch_vggblstmp_e8/decode_train_dev_beam20_eacc.best_p0.3_len0.0-0.8/result.txt:| Sum/Avg                           | 3999    47130 | 69.4    23.1    7.4     3.3   33.9    78.1 |
```

# first result
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_sp_a01/decode_*_beam20_eacc.best_p0_len0.0-0.8/result.txt
exp/train_nodup_sp_a01/decode_dev_beam20_eacc.best_p0_len0.0-0.8/result.txt:| SPKR                              | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/train_nodup_sp_a01/decode_dev_beam20_eacc.best_p0_len0.0-0.8/result.txt:| Sum/Avg                           | 5413   56154 | 66.9   21.0   12.1    2.5   35.6   75.0 |
exp/train_nodup_sp_a01/decode_train_dev_beam20_eacc.best_p0_len0.0-0.8/result.txt:| SPKR                              | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/train_nodup_sp_a01/decode_train_dev_beam20_eacc.best_p0_len0.0-0.8/result.txt:| Sum/Avg                           | 3999   47130 | 62.6   20.6   16.8    2.5   39.9   78.7 |
```
