# Transformer 300 epochs, decoder 6 layer 2048 unitsns
  - Environments (obtained by `$ get_sys_info.sh`)
      - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
	  - python version: `Python 3.7.3`
	  - espnet version: `espnet 0.3.1`
	  - chainer version: `chainer 6.0.0`
	  - pytorch version: `pytorch 1.0.1.post2`
	  - Git hash: `2525193c2c25dea5683086ef1b69f45bd1e050af`
  - It takes a very long time for the decoding and I don't recommend to use this setup without speed improvement during decoding
  - Model files (archived to v2.tgz by `$ pack_model.sh`)
      - model link: https://drive.google.com/open?id=1qtCKSHLZdGdLaZVZ-33NoX3nf_L3Pto8
	  - training config file: `conf/tuning/train_pytorch_transformer_d6-2048.yaml`
	  - decoding conf-if file: `conf/tuning/decode_pytorch_transformer.yaml`
      - cmvn file: `data/tr_it/cmvn.ark`
      - e2e file: `exp/tr_it_pytorch_train_d6-2048/results/model.last10.avg.best`
      - e2e JSON file: `exp/tr_it_pytorch_train_d6-2048/results/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/tr_it_pytorch_train_d6-2048/decode_dt_it_decode/result.txt
| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1082   79133 | 92.5    3.8    3.7    1.9    9.4   95.0 |
exp/tr_it_pytorch_train_d6-2048/decode_et_it_decode/result.txt
| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1055   77966 | 92.6    3.7    3.7    1.7    9.1   95.6 |
```

# Transformer 300 epochs, decoder 1 layer 1024 units 
  - Environments (obtained by `$ get_sys_info.sh`)
      - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
	  - python version: `Python 3.7.3`
	  - espnet version: `espnet 0.3.1`
	  - chainer version: `chainer 6.0.0`
	  - pytorch version: `pytorch 1.0.1.post2`
	  - Git hash: `2525193c2c25dea5683086ef1b69f45bd1e050af`
  - Model files (archived to v1.tgz by `$ pack_model.sh`)
      - model link: https://drive.google.com/open?id=1xgTpS7dqcZs75CvH4OihbXdAJtOM43Fe
      - training config file: `conf/tuning/train_pytorch_transformer.yaml`
      - decoding conf-if file: `conf/tuning/decode_pytorch_transformer.yaml`
      - cmvn file: `data/tr_it/cmvn.ark`
      - e2e file: `exp/tr_it_pytorch_train/results/model.last10.avg.best`
      - e2e JSON file: `exp/tr_it_pytorch_train/results/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/tr_it_pytorch_train/decode_dt_it_decode/result.txt
| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1082   79133 | 92.0    3.9    4.1    1.8    9.8   96.7 |
exp/tr_it_pytorch_train/decode_et_it_decode/result.txt
| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1055   77966 | 92.0    3.9    4.0    1.7    9.6   96.7 |
```

# Transformer 100 epochs
```bash
shinji@b14:/export/a08/shinji/201707e2e/espnet_dev6/egs/voxforge/asr2$ grep -e Avg -e SPKR -m 2 exp/tr_it_pytorch_nopatience/decode_dt_it_decode/result.txt
| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1082   79133 | 90.1    4.2    5.7    2.3   12.2   98.6 |
shinji@b14:/export/a08/shinji/201707e2e/espnet_dev6/egs/voxforge/asr2$ grep -e Avg -e SPKR -m 2 exp/tr_it_pytorch_nopatience/decode_et_it_decode/result.txt
| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg               | 1055   77966 | 89.7    4.4    6.0    2.2   12.5   99.1 |
```

# RNN default
- change several update including ctc/attention decoding, label smoothing, and fixed search parameters
```bash
write a CER (or TER) result in exp/tr_it_debug_alpha0.5/decode_dt_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.5/result.txt
| SPKR                  | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
| Sum/Avg               | 1082    79133 | 89.6     5.5    5.0     2.5   12.9    98.2 |
write a CER (or TER) result in exp/tr_it_debug_alpha0.5/decode_et_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.5/result.txt
| SPKR                  | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
| Sum/Avg               | 1055    77966 | 89.7     5.5    4.8     2.3   12.6    98.4 |
```

# Scheduled sampling experiments by enabling mtlalpha=0.0 and scheduled-sampling-ratio with 0.0 and 0.5
- Number of decoder layers = 1
```bash
exp/tr_it_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampratio0.0_bs30_mli800_mlo150_epochs30/decode_et_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.0/result.txt:
|        SPKR                         |         # Snt                 # Wrd         |         Corr                   Sub                    Del                   Ins                    Err                 S.Err         |
|        Sum/Avg                      |          895                  66163         |         29.4                  21.5                   49.2                   4.2                   74.8                 100.0         |
exp/tr_it_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampratio0.5_bs30_mli800_mlo150_epochs30/decode_et_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.0/result.txt:
|        SPKR                         |         # Snt                 # Wrd         |         Corr                   Sub                    Del                   Ins                    Err                 S.Err         |
|        Sum/Avg                      |          895                  66163         |         88.0                   6.7                    5.3                   3.0                   15.0                  98.7         |
```
- Number of decoder layers = 2
```bash
exp/tr_it_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d2_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampratio0.0_bs30_mli800_mlo150_epochs30/decode_et_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.0/result.txt:
|        SPKR                         |         # Snt                 # Wrd         |         Corr                   Sub                    Del                   Ins                    Err                 S.Err         |
|        Sum/Avg                     |           895                  66163        |          30.7                   22.1                  47.2                   3.9                   73.2                 100.0         |
exp/tr_it_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d2_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_sampratio0.5_bs30_mli800_mlo150_epochs30/decode_et_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.0/result.txt:
|        SPKR                         |         # Snt                 # Wrd         |         Corr                   Sub                    Del                   Ins                    Err                 S.Err         |
|        Sum/Avg                      |          895                  66163         |         36.4                   30.3                   33.4                  9.1                   72.8                 100.0         |
```

# Change several update including ctc/attention decoding, label smoothing, and fixed search parameters
```bash
write a CER (or TER) result in exp/tr_it_debug_alpha0.5/decode_dt_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.5/result.txt
| SPKR                  | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
| Sum/Avg               | 1082    79133 | 89.6     5.5    5.0     2.5   12.9    98.2 |
write a CER (or TER) result in exp/tr_it_debug_alpha0.5/decode_et_it_beam20_eacc.best_p0_len0.0-0.0_ctcw0.5/result.txt
| SPKR                  | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
| Sum/Avg               | 1055    77966 | 89.7     5.5    4.8     2.3   12.6    98.4 |
```

# change minlenratio from 0.0 to 0.2
```bash
exp/tr_it_d1_debug_chainer/decode_dt_it_beam20_eacc.best_p0_len0.2-0.8/result.txt:| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/tr_it_d1_debug_chainer/decode_dt_it_beam20_eacc.best_p0_len0.2-0.8/result.txt:| Sum/Avg               | 1082   79133 | 88.3    6.1    5.6    3.2   14.9   98.9 |
exp/tr_it_d1_debug_chainer/decode_et_it_beam20_eacc.best_p0_len0.2-0.8/result.txt:| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/tr_it_d1_debug_chainer/decode_et_it_beam20_eacc.best_p0_len0.2-0.8/result.txt:| Sum/Avg               | 1055   77966 | 88.4    6.0    5.6    2.9   14.5   98.9 |
```

# Change NStepLSTM to StatelessLSTM
```bash
$ grep -e Avg -e SPKR -m 2 exp/tr_it_a02/decode_*t_it_beam20_eacc.best_p0_len0.0-0.8/result.txt
exp/tr_it_a02/decode_dt_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/tr_it_a02/decode_dt_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| Sum/Avg               | 1080   78951 | 87.7    5.7    6.6    2.9   15.2   97.7 |
exp/tr_it_a02/decode_et_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/tr_it_a02/decode_et_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| Sum/Avg               | 1050   77586 | 87.3    5.8    6.9    2.8   15.5   97.5 |
```

# VGGBLSMP, adaeldta with eps decay monitoring validation accuracy
```bash
$ grep Avg exp/tr_it_a10/decode_*t_it_beam20_eacc.best_p0_len0.0-0.8/result.txt 
exp/tr_it_a10/decode_dt_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/tr_it_a10/decode_dt_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| Sum/Avg               | 1080   78951 | 86.7    5.9    7.3    3.2   16.5   98.1 |
exp/tr_it_a10/decode_et_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| SPKR                  | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
exp/tr_it_a10/decode_et_it_beam20_eacc.best_p0_len0.0-0.8/result.txt:| Sum/Avg               | 1050   77586 | 86.3    5.6    8.1    2.8   16.5   98.3 |
```

