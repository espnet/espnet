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

# Transducer

## Summary

|Model|Algo|CER¹|WER¹|SER¹|RTF¹²|
|-|-|-|-|-|-|
|RNN-T|default|12.3|42.5|96.6|0.097|
|-|ALSD|12.2|42.9|96.6|0.083|
|-|TSD|12.0|42.0|96.4|0.139|
|-|NSC|12.0|42.3|96.4|0.156|
|-|mAES|12.1|42.3|96.6|0.075|
|RNN-T + Aux|default|11.6|40.6|95.5|0.098|
|-|ALSD|11.5|40.5|95.6|0.082|
|-|TSD|11.3|39.7|94.9|0.140|
|-|NSC|11.3|40.0|95.3|0.156|
|-|mAES|11.5|40.2|95.2|0.076|
|Conformer/RNN-T|default|8.8|32.6|92.6|0.137|
|-|ALSD|8.7|32.6|92.4|0.151|
|-|TSD|8.8|32.8|92.8|0.298|
|-|NSC|8.9|33.1|93.1|0.325|
|-|mAES|8.7|32.8|92.8|0.108|
|Conformer/RNN-T + Aux|default|7.9|28.7|88.0|0.159|
|-|ALSD|7.9|28.7|88.8|0.146|
|-|TSD³|7.8|28.9|88.6|0.202|
|-|NSC³|7.8|29.0|88.6|0.224|
|-|mAES|7.8|28.9|88.7|0.109|

¹ Reported on the test set only.
² RTF was computed using `line-profiler` tool applied to [recognize method](https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr_transducer.py#L470). The reported value is averaged on 5 runs with `nj=1`. All experiments were performed using a single AMD EPYC 7502P.
³ Here, the number of required expansions at each timestep for the time-synchronous algorithms can be lowered with almost no degradation in terms of CER/WER. Because of its adaptive nature, mAES will automatically adjusts the number of required expansions at each time step.
  Thus, we use `max-sym-exp: 3` for TSD and `nstep: 2` for NSC when decoding with Conformer/RNN-T model trained with aux. tasks.

## RNN-Transducer (Enc: VGG + 4x BLSTM, Dec: 1x LSTM)

- General information
  - GPU: Nvidia A100 40Gb
  - Peak VRAM usage during training: ~ 18.1 GiB
  - Training time: ~ 21 minutes
  - Decoding time (16 jobs, `search-type: default`): ~ 53 seconds

- Environments
  - date: `Wed Aug 18 07:56:16 UTC 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1mMAqbmF-GgTCWFwl3EXunpLTdWGQQmyo
  - training config file: `conf/tuning/transducer/train_rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/tr_it/cmvn.ark`
  - e2e file: `exp/tr_it_pytorch_train_rnn_transducer/results/model.loss.best`
  - e2e JSON file: `exp/tr_it_pytorch_train_rnn_transducer/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|75494|89.6|5.0|5.4|2.6|13.0|98.2|
|decode_dt_it_decode_default|1035|75494|89.5|5.0|5.5|2.6|13.1|98.4|
|decode_dt_it_decode_maes|1035|75494|89.3|5.0|5.6|2.4|13.1|98.1|
|decode_dt_it_decode_nsc|1035|75494|89.4|5.0|5.6|2.4|13.0|98.5|
|decode_dt_it_decode_tsd|1035|75494|89.4|4.9|5.6|2.4|12.9|98.4|
|decode_et_it_decode_alsd|1103|81228|90.4|4.9|4.6|2.6|12.2|96.6|
|decode_et_it_decode_default|1103|81228|90.4|4.9|4.7|2.7|12.2|96.6|
|decode_et_it_decode_maes|1103|81228|90.4|4.8|4.8|2.5|12.1|96.6|
|decode_et_it_decode_nsc|1103|81228|90.5|4.8|4.7|2.5|12.0|96.4|
|decode_et_it_decode_tsd|1103|81228|90.5|4.7|4.8|2.4|12.0|96.4|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|12587|60.2|32.7|7.1|4.5|44.3|98.2|
|decode_dt_it_decode_default|1035|12587|60.2|32.7|7.1|4.5|44.4|98.4|
|decode_dt_it_decode_maes|1035|12587|60.1|32.6|7.3|4.3|44.2|98.1|
|decode_dt_it_decode_nsc|1035|12587|60.1|32.5|7.4|4.3|44.1|98.5|
|decode_dt_it_decode_tsd|1035|12587|60.4|32.2|7.4|4.1|43.8|98.4|
|decode_et_it_decode_alsd|1103|13699|62.0|31.8|6.1|5.0|42.9|96.6|
|decode_et_it_decode_default|1103|13699|62.5|31.4|6.1|4.9|42.5|96.6|
|decode_et_it_decode_maes|1103|13699|62.2|31.4|6.3|4.5|42.3|96.6|
|decode_et_it_decode_nsc|1103|13699|62.3|31.5|6.2|4.6|42.3|96.4|
|decode_et_it_decode_tsd|1103|13699|62.4|31.3|6.3|4.4|42.0|96.4|

## RNN-Transducer (Enc: VGG + 4x BLSTM, Dec: 1x LSTM)
##   + CTC loss + Label Smoothing loss + aux. Transducer loss + symm. KL div loss

- General information
  - GPU: Nvidia A100 40Gb
  - Peak VRAM usage during training: ~ 18.4 GiB
  - Training time: ~ 25 minutes and 20 seconds
  - Decoding time (16 jobs, `search-type: default`): ~ 55 seconds

- Environments
  - date: `Wed Aug 18 07:56:16 UTC 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1ZW_wSJYEiXyp0n_Cj09ucgh2ncZ6-RK1
  - training config file: `conf/tuning/transducer/train_rnn_transducer_aux.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/tr_it/cmvn.ark`
  - e2e file: `exp/tr_it_pytorch_train_rnn_transducer_aux/results/model.loss.best`
  - e2e JSON file: `exp/tr_it_pytorch_train_rnn_transducer_aux/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|75494|90.2|4.7|5.2|2.3|12.1|98.6|
|decode_dt_it_decode_default|1035|75494|89.9|4.6|5.5|2.2|12.3|97.7|
|decode_dt_it_decode_maes|1035|75494|89.9|4.7|5.4|2.2|12.3|97.9|
|decode_dt_it_decode_nsc|1035|75494|90.0|4.6|5.4|2.1|12.1|97.8|
|decode_dt_it_decode_tsd|1035|75494|90.0|4.6|5.4|2.1|12.1|97.4|
|decode_et_it_decode_alsd|1103|81228|90.8|4.6|4.5|2.3|11.5|95.6|
|decode_et_it_decode_default|1103|81228|90.6|4.6|4.8|2.2|11.6|95.5|
|decode_et_it_decode_maes|1103|81228|90.7|4.6|4.7|2.1|11.5|95.2|
|decode_et_it_decode_nsc|1103|81228|90.8|4.5|4.7|2.1|11.3|95.3|
|decode_et_it_decode_tsd|1103|81228|90.8|4.5|4.7|2.1|11.3|94.9|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|12587|62.0|30.9|7.1|3.8|41.8|98.6|
|decode_dt_it_decode_default|1035|12587|61.5|31.0|7.5|3.6|42.1|97.7|
|decode_dt_it_decode_maes|1035|12587|61.7|30.9|7.4|3.4|41.7|97.9|
|decode_dt_it_decode_nsc|1035|12587|61.8|30.7|7.5|3.4|41.6|97.8|
|decode_dt_it_decode_tsd|1035|12587|62.1|30.6|7.4|3.5|41.4|97.4|
|decode_et_it_decode_alsd|1103|13699|63.5|30.3|6.2|4.0|40.5|95.6|
|decode_et_it_decode_default|1103|13699|63.3|30.6|6.2|3.9|40.6|95.5|
|decode_et_it_decode_maes|1103|13699|63.4|30.2|6.3|3.6|40.2|95.2|
|decode_et_it_decode_nsc|1103|13699|63.7|29.9|6.3|3.7|40.0|95.3|
|decode_et_it_decode_tsd|1103|13699|64.0|29.7|6.3|3.7|39.7|94.9|

## Conformer/RNN-Transducer (Enc: VGG + 8x Conformer, Dec: 1x LSTM)

- General information
  - GPU: Nvidia A100 40Gb
  - Peak VRAM usage during training: ~ 24.8 GiB
  - Training time: ~ 4 hours and 13 minutes
  - Decoding time (16 jobs, `search-type: default`): ~ 2 minutes and 30 seconds
  - Model averaging: `n_average=20`, `use_valbest_average=false`

- Environments
  - date: `Wed Aug 18 07:56:16 UTC 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1JLy91GORrg-iw_ZmRfevGaLjMZ_NmLa6
  - training config file: `conf/tuning/transducer/train_conformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/tr_it/cmvn.ark`
  - e2e file: `exp/tr_it_pytorch_train_conformer-rnn_transducer/results/model.last20.avg.best`
  - e2e JSON file: `exp/tr_it_pytorch_train_conformer-rnn_transducer/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|75494|92.7|3.6|3.7|2.2|9.5|94.5|
|decode_dt_it_decode_default|1035|75494|92.7|3.6|3.7|2.2|9.5|94.8|
|decode_dt_it_decode_maes|1035|75494|92.6|3.6|3.8|2.1|9.5|95.1|
|decode_dt_it_decode_nsc|1035|75494|92.5|3.7|3.9|2.1|9.7|95.2|
|decode_dt_it_decode_tsd|1035|75494|92.4|3.6|3.9|2.1|9.6|94.9|
|decode_et_it_decode_alsd|1103|81228|93.4|3.5|3.0|2.2|8.7|92.4|
|decode_et_it_decode_default|1103|81228|93.4|3.5|3.0|2.2|8.8|92.6|
|decode_et_it_decode_maes|1103|81228|93.3|3.5|3.1|2.1|8.7|92.8|
|decode_et_it_decode_nsc|1103|81228|93.2|3.6|3.2|2.1|8.9|93.1|
|decode_et_it_decode_tsd|1103|81228|93.2|3.5|3.2|2.0|8.8|92.8|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|12587|69.9|24.2|5.8|4.1|34.2|94.5|
|decode_dt_it_decode_default|1035|12587|70.0|24.1|5.9|4.2|34.1|94.8|
|decode_dt_it_decode_maes|1035|12587|69.6|24.4|6.1|4.0|34.4|95.1|
|decode_dt_it_decode_nsc|1035|12587|69.2|24.4|6.4|4.1|34.9|95.2|
|decode_dt_it_decode_tsd|1035|12587|69.2|24.4|6.4|3.9|34.7|94.9|
|decode_et_it_decode_alsd|1103|13699|71.5|23.4|5.1|4.1|32.6|92.4|
|decode_et_it_decode_default|1103|13699|71.4|23.7|4.9|4.0|32.6|92.6|
|decode_et_it_decode_maes|1103|13699|71.1|23.7|5.2|3.9|32.8|92.8|
|decode_et_it_decode_nsc|1103|13699|70.7|24.0|5.3|3.9|33.2|93.1|
|decode_et_it_decode_tsd|1103|13699|70.9|23.7|5.4|3.7|32.8|92.8|

## Conformer/RNN-Transducer (Enc: VGG + 8x Conformer, Dec: 1x LSTM)
##   + CTC loss + Label Smoothing loss + aux. Transducer loss + symm. KL div loss

- General information
  - GPU: Nvidia A100 40Gb
  - Peak VRAM usage during training: ~ 26.9 GiB
  - Training time: ~ 4 hours and 31 minutes
  - Decoding time (16 jobs, `search-type: default`): ~ 2 minutes and 50 seconds
  - Model averaging: `n_average=20`, `use_valbest_average=false`

- Environments
  - date: `Wed Aug 18 07:56:16 UTC 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1imTL8FyhmIO5OmSl1h-D91UwI6S0elx-
  - training config file: `conf/tuning/transducer/train_conformer-rnn_transducer_aux.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/tr_it/cmvn.ark`
  - e2e file: `exp/tr_it_pytorch_train_conformer-rnn_transducer_aux/results/model.last20.avg.best`
  - e2e JSON file: `exp/tr_it_pytorch_train_conformer-rnn_transducer_aux/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|75494|93.1|3.4|3.5|1.9|8.8|93.6|
|decode_dt_it_decode_default|1035|75494|93.1|3.5|3.5|1.9|8.8|93.3|
|decode_dt_it_decode_maes|1035|75494|93.0|3.4|3.5|1.8|8.8|93.2|
|decode_dt_it_decode_nsc|1035|75494|93.1|3.5|3.5|1.8|8.8|93.5|
|decode_dt_it_decode_tsd|1035|75494|93.0|3.4|3.5|1.8|8.8|93.4|
|decode_et_it_decode_alsd|1103|81228|94.0|3.2|2.8|1.9|7.9|88.8|
|decode_et_it_decode_default|1103|81228|94.0|3.2|2.8|1.9|7.9|88.0|
|decode_et_it_decode_maes|1103|81228|93.9|3.2|3.0|1.7|7.9|88.2|
|decode_et_it_decode_nsc|1103|81228|93.9|3.2|3.0|1.7|7.8|88.6|
|decode_et_it_decode_tsd|1103|81228|93.9|3.2|3.0|1.7|7.8|88.6|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_it_decode_alsd|1035|12587|71.8|22.7|5.5|3.5|31.7|93.6|
|decode_dt_it_decode_default|1035|12587|71.7|22.9|5.4|3.4|31.7|93.3|
|decode_dt_it_decode_maes|1035|12587|71.7|22.8|5.5|3.3|31.6|93.2|
|decode_dt_it_decode_nsc|1035|12587|71.6|22.9|5.5|3.3|31.6|93.5|
|decode_dt_it_decode_tsd|1035|12587|71.8|22.7|5.5|3.3|31.5|93.4|
|decode_et_it_decode_alsd|1103|13699|74.3|21.4|4.2|3.1|28.7|88.8|
|decode_et_it_decode_default|1103|13699|74.3|21.3|4.3|3.1|28.7|88.0|
|decode_et_it_decode_maes|1103|13699|73.9|21.5|4.6|2.8|28.9|88.7|
|decode_et_it_decode_nsc|1103|13699|73.9|21.5|4.6|2.9|29.0|88.6|
|decode_et_it_decode_tsd|1103|13699|73.9|21.4|4.7|2.8|28.9|88.6|
