# Conformer results

## Environments
- date: `Fri Nov  6 18:57:07 JST 2020`
- python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.1.0`
- Git hash: `87bc8f1a1ad548b7dfc062c0820c969200fed3b4`
  - Commit date: `Sat Sep 5 01:30:59 2020 +0900`
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1-s4AUuqCyNNH-lZk9ZLLWH11cdDSLnL5/view?usp=sharing
  - training config file: `conf/work/train_pytorch_conformer_k7.yaml`
  - decoding config file: `conf/work/decode_pytorch_transformer_bs20_ctc0.3.yaml`
  - cmvn file: `data/tr_it/cmvn.ark`
  - e2e file: `exp/tr_it_pytorch_train_pytorch_conformer_k7/results/model.last10.avg.best`
  - e2e JSON file: `exp/tr_it_pytorch_train_pytorch_conformer_k7/results/model.json`

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_pytorch_transformer_bs1_ctc0.3|1082|13235|71.2|24.1|4.7|3.8|32.6|94.4|
|decode_dt_it_decode_pytorch_transformer_bs20_ctc0.3|1082|13235|71.1|24.4|4.6|3.7|32.6|94.2|
|decode_et_it_decode_pytorch_transformer_bs1_ctc0.3|1055|12990|73.4|22.2|4.4|3.9|30.5|92.8|
|decode_et_it_decode_pytorch_transformer_bs20_ctc0.3|1055|12990|73.4|22.2|4.3|3.9|30.5|93.1|
```

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_pytorch_transformer_bs1_ctc0.3|1082|79133|92.9|3.8|3.3|2.0|9.1|93.5|
|decode_dt_it_decode_pytorch_transformer_bs20_ctc0.3|1082|79133|93.2|3.8|3.0|1.9|8.7|94.1|
|decode_et_it_decode_pytorch_transformer_bs1_ctc0.3|1055|77966|93.3|3.4|3.3|1.7|8.4|92.3|
|decode_et_it_decode_pytorch_transformer_bs20_ctc0.3|1055|77966|93.5|3.4|3.1|1.7|8.2|92.4|
```

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
      - model link: https://drive.google.com/open?id=1xdNm0-SFcZHwHdYX3-O395-9bNFf-w5G
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
      - model link: https://drive.google.com/open?id=1b_dVbjh4H0Tfi2ZlCvG8RnQ8FPoVbfHc
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

# Below are preliminaries results for transducer and transducer-attention.


# RNN-Transducer ('rnnt')

- Environments
  - date: `Mon Jul 13 11:18:28 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `c1a32dab8d3b5d1e213e1e74c0a1f355b2adf6f5`
  - Commit date: `Sun Jul 12 13:46:35 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|79133|90.3|4.8|4.9|2.6|12.3|97.0|
|decode_dt_it_decode_default|1082|79133|89.5|4.8|5.6|2.5|13.0|97.4|
|decode_dt_it_decode_nsc|1082|79133|90.3|4.7|5.0|2.5|12.2|97.2|
|decode_dt_it_decode_tsd|1082|79133|90.2|4.6|5.2|2.5|12.3|97.0|
|decode_et_it_decode_alsd|1055|77966|90.1|5.0|4.8|2.7|12.5|97.7|
|decode_et_it_decode_default|1055|77966|89.6|5.0|5.4|2.3|12.8|98.2|
|decode_et_it_decode_nsc|1055|77966|90.2|4.9|4.8|2.4|12.2|97.2|
|decode_et_it_decode_tsd|1055|77966|90.1|4.9|5.0|2.4|12.3|97.2|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|13235|62.7|31.3|5.9|4.9|42.1|97.0|
|decode_dt_it_decode_default|1082|13235|61.0|32.3|6.7|4.5|43.5|97.4|
|decode_dt_it_decode_nsc|1082|13235|62.9|31.0|6.1|4.7|41.9|97.2|
|decode_dt_it_decode_tsd|1082|13235|62.7|30.9|6.4|4.5|41.8|97.0|
|decode_et_it_decode_alsd|1055|12990|61.9|31.7|6.5|5.9|44.0|97.7|
|decode_et_it_decode_default|1055|12990|60.7|32.3|7.0|5.3|44.6|98.2|
|decode_et_it_decode_nsc|1055|12990|62.4|31.2|6.4|5.2|42.8|97.2|
|decode_et_it_decode_tsd|1055|12990|62.1|31.3|6.5|5.0|42.9|97.2|

# RNN-Transducer + encoder pre-initialization (CTC)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|79133|90.9|4.8|4.3|2.6|11.7|97.0|
|decode_dt_it_decode_default|1082|79133|90.7|4.7|4.6|2.5|11.8|97.5|
|decode_dt_it_decode_nsc|1082|79133|90.9|4.7|4.4|2.5|11.6|97.1|
|decode_dt_it_decode_tsd|1082|79133|90.8|4.7|4.5|2.4|11.6|97.2|
|decode_et_it_decode_alsd|1055|77966|91.0|4.8|4.2|2.4|11.4|97.8|
|decode_et_it_decode_default|1055|77966|90.8|4.8|4.4|2.2|11.4|97.7|
|decode_et_it_decode_nsc|1055|77966|91.0|4.7|4.2|2.3|11.3|97.3|
|decode_et_it_decode_tsd|1055|77966|91.0|4.7|4.3|2.2|11.3|97.3|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|13235|61.9|32.4|5.7|4.9|42.9|97.0|
|decode_dt_it_decode_default|1082|13235|61.5|32.5|6.0|4.5|43.1|97.5|
|decode_dt_it_decode_nsc|1082|13235|61.9|32.2|5.9|4.6|42.7|97.1|
|decode_dt_it_decode_tsd|1082|13235|61.9|32.2|5.9|4.5|42.6|97.2|
|decode_et_it_decode_alsd|1055|12990|61.5|33.1|5.5|4.7|43.3|97.8|
|decode_et_it_decode_default|1055|12990|61.0|33.0|6.1|4.3|43.3|97.7|
|decode_et_it_decode_nsc|1055|12990|61.6|32.8|5.6|4.5|43.0|97.3|
|decode_et_it_decode_tsd|1055|12990|61.5|32.7|5.8|4.4|42.9|97.3|

# RNN-Transducer + encoder pre-initialization (CTC) + decoder pre-initialization (LM)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|79133|90.9|4.7|4.4|2.5|11.6|97.3|
|decode_dt_it_decode_default|1082|79133|90.6|4.7|4.7|2.2|11.6|97.1|
|decode_dt_it_decode_nsc|1082|79133|90.8|4.7|4.5|2.3|11.5|96.9|
|decode_dt_it_decode_tsd|1082|79133|90.7|4.7|4.6|2.2|11.5|96.9|
|decode_et_it_decode_alsd|1055|77966|90.9|4.8|4.3|2.3|11.4|97.5|
|decode_et_it_decode_default|1055|77966|90.7|4.7|4.6|2.2|11.5|97.5|
|decode_et_it_decode_nsc|1055|77966|90.9|4.7|4.4|2.2|11.3|97.5|
|decode_et_it_decode_tsd|1055|77966|90.8|4.7|4.5|2.2|11.4|97.6|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|13235|62.0|32.2|5.8|4.5|42.5|97.3|
|decode_dt_it_decode_default|1082|13235|61.2|32.2|6.6|3.9|42.7|97.1|
|decode_dt_it_decode_nsc|1082|13235|61.9|32.0|6.2|4.1|42.2|96.9|
|decode_dt_it_decode_tsd|1082|13235|61.8|31.9|6.3|4.0|42.2|96.9|
|decode_et_it_decode_alsd|1055|12990|61.5|32.5|6.0|4.7|43.2|97.5|
|decode_et_it_decode_default|1055|12990|61.2|32.4|6.4|4.2|43.0|97.5|
|decode_et_it_decode_nsc|1055|12990|61.7|32.2|6.1|4.4|42.7|97.5|
|decode_et_it_decode_tsd|1055|12990|61.6|32.2|6.2|4.3|42.7|97.6|

# Conformer/RNN-Transducer (enc: 8 x Conformer, dec: 1 x LSTM)
# modified decoding params:
#   - general: n_average=20
#   - alsd: beam-size=10, u-max=300, score-norm=True
#   - nsc: nstep=4, prefix-alpha=3
#   - tsd: max-sym-exp=5, score-norm=True

- Environments
  - date: `Fri Nov 27 11:41:31 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `b0ec37da0357c3b612833b02b45994eaaa4370ae`
  - Commit date: `Fri Nov 27 10:14:20 2020 +0100

- Model files (archived to conformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=13HHW3Qs5Yk4vzmEuOHl-VJ46M-e3qtZ6
  - training config file: `conf/tuning/transducer/train_conformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/tr_it/cmvn.ark`
  - e2e file: `exp/tr_it_pytorch_train_conformer-rnn_transducer/results/model.last20.avg.best`
  - e2e JSON file: `exp/tr_it_pytorch_train_conformer-rnn_transducer/results/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|79133|93.6|3.5|2.9|2.4|8.8|93.5|
|decode_dt_it_decode_default|1082|79133|93.6|3.4|3.0|2.4|8.8|93.3|
|decode_dt_it_decode_nsc|1082|79133|93.6|3.4|3.0|2.3|8.7|93.2|
|decode_dt_it_decode_tsd|1082|79133|93.6|3.4|3.0|2.4|8.8|93.3|
|decode_et_it_decode_alsd|1055|77966|93.4|3.5|3.1|2.1|8.6|93.3|
|decode_et_it_decode_default|1055|77966|93.5|3.5|3.1|2.1|8.7|92.9|
|decode_et_it_decode_nsc|1055|77966|93.4|3.5|3.1|2.1|8.7|93.3|
|decode_et_it_decode_tsd|1055|77966|93.4|3.5|3.1|2.1|8.7|93.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1082|13235|71.5|23.9|4.6|4.1|32.7|93.5|
|decode_dt_it_decode_default|1082|13235|71.4|23.9|4.7|4.3|32.9|93.3|
|decode_dt_it_decode_nsc|1082|13235|71.4|23.9|4.7|4.2|32.9|93.2|
|decode_dt_it_decode_tsd|1082|13235|71.4|23.9|4.7|4.3|32.9|93.3|
|decode_et_it_decode_alsd|1055|12990|71.8|23.6|4.6|4.2|32.4|93.3|
|decode_et_it_decode_default|1055|12990|71.8|23.6|4.6|4.4|32.6|92.9|
|decode_et_it_decode_nsc|1055|12990|71.7|23.6|4.7|4.2|32.5|93.3|
|decode_et_it_decode_tsd|1055|12990|71.7|23.7|4.6|4.2|32.5|93.0|