# conformer with BPE 2000, specaug, speed perturbation, Transformer LM decoding
## Models
- model link: https://drive.google.com/file/d/1FhDeQ4eFxBsnGitZkG7rgScZB0oNpeg7/view
- training config file: `conf/tuning/train_pytorch_conformer_lr5.yaml`
- preprocess config file: `conf/specaug.yaml`
- decoding config file: `conf/decode.yaml`

## WER
```
exp_sp/train_nodup_sp_pytorch_train_pytorch_conformer_lr5_specaug_resume/decode_eval2000_model.last10.avg.best_decode_train_transformer_lm_pytorch_swbd+fisher_bpe2000/scoring/hyp.callhm.ctm.filt.sys
|       SPKR              |        # Snt              # Wrd        |        Corr                 Sub                  Del                 Ins                  Err               S.Err        |
|       Sum/Avg           |        2628               21594        |        87.9                 8.9                  3.2                 2.0                 14.0                49.8        |
exp_sp/train_nodup_sp_pytorch_train_pytorch_conformer_lr5_specaug_resume/decode_eval2000_model.last10.avg.best_decode_train_transformer_lm_pytorch_swbd+fisher_bpe2000/scoring/hyp.ctm.filt.sys
|       SPKR              |       # Snt             # Wrd        |       Corr                 Sub                Del                 Ins                 Err              S.Err        |
|       Sum/Avg           |       4459              42989        |       91.0                 6.5                2.5                 1.4                10.4               44.5        |
exp_sp/train_nodup_sp_pytorch_train_pytorch_conformer_lr5_specaug_resume/decode_eval2000_model.last10.avg.best_decode_train_transformer_lm_pytorch_swbd+fisher_bpe2000/scoring/hyp.swbd.ctm.filt.sys
|       SPKR             |        # Snt              # Wrd        |       Corr                  Sub                 Del                 Ins                  Err               S.Err        |
|       Sum/Avg          |        1831               21395        |       94.1                  4.1                 1.9                 0.9                  6.8                36.9        |
```

# transformer with BPE 2000, specaug, LM decoding
## Models
- model link: https://drive.google.com/open?id=10AeST49tCFOHQul4rVETBHISwaN4PwuU
- training config file: `conf/tuning/train_pytorch_transformer.yaml`
- decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
- cmvn file: `data/train_nodup/cmvn.ark`
- e2e file: `exp/train_nodup_pytorch_train_specaug/results/model.last10.avg.best`
- e2e JSON file: `exp/train_nodup_pytorch_train_specaug/results/model.json`
- lm file: `exp/train_rnnlm_pytorch_swbd+fisher_bpe2000/rnnlm.model.best`
- lm JSON file: `exp/train_rnnlm_pytorch_swbd+fisher_bpe2000/model.json`
- dict file: `data/lang_char`

## Environments
- date: `Tue Dec 24 12:01:51 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `fb9911bef3dcc6266c8fe4bd25f9bb5bbe40fb89`
  - Commit date: `Tue Dec 24 11:08:47 2019 +0900`

## WER
```
exp/train_nodup_pytorch_train_specaug/decode_eval2000_model.last10.avg.best_decode_lmw0.3_swbd+fisher/scoring/hyp.callhm.ctm.filt.sys:|    SPK          |    # Snt       # Wrd     |    Corr          Sub          Del          Ins           Err        S.Err    |
exp/train_nodup_pytorch_train_specaug/decode_eval2000_model.last10.avg.best_decode_lmw0.3_swbd+fisher/scoring/hyp.callhm.ctm.filt.sys:|    SumAvg       |    2628        21594     |    85.2         11.2          3.6          2.6          17.3         53.7    |
exp/train_nodup_pytorch_train_specaug/decode_eval2000_model.last10.avg.best_decode_lmw0.3_swbd+fisher/scoring/hyp.ctm.filt.sys:|   SPKR          |    # Snt      # Wrd    |    Corr         Sub          Del         Ins          Err       S.Err    |
exp/train_nodup_pytorch_train_specaug/decode_eval2000_model.last10.avg.best_decode_lmw0.3_swbd+fisher/scoring/hyp.ctm.filt.sys:|   Sum/Avg       |    4459       42989    |    88.9         8.2          2.9         1.8         12.9        48.0    |
exp/train_nodup_pytorch_train_specaug/decode_eval2000_model.last10.avg.best_decode_lmw0.3_swbd+fisher/scoring/hyp.swbd.ctm.filt.sys:|    SPKR         |    # Snt       # Wrd    |    Corr          Sub           Del          Ins          Err        S.Err    |
exp/train_nodup_pytorch_train_specaug/decode_eval2000_model.last10.avg.best_decode_lmw0.3_swbd+fisher/scoring/hyp.swbd.ctm.filt.sys:|    Sum/Avg      |    1831        21395    |    92.5          5.3           2.2          1.0          8.5         40.0    |
```

# transformer with BPE 2000, specaug

```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/*.sys
exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/hyp.callhm.ctm.filt.sys:|   SPKR         |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err       S.Err   |
exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/hyp.callhm.ctm.filt.sys:|   Sum/Avg      |   2628       21594   |   84.6       12.0         3.5        2.7       18.1        55.7   |
exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/hyp.ctm.filt.sys:|   SPKR         |  # Snt     # Wrd   |  Corr        Sub       Del        Ins       Err      S.Err   |
exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/hyp.ctm.filt.sys:|   Sum/Avg      |  4459      42989   |  88.4        8.8       2.8        1.9      13.6       50.1   |
exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/hyp.swbd.ctm.filt.sys:|   SPKR        |   # Snt     # Wrd    |   Corr        Sub        Del        Ins         Err      S.Err   |
exp/train_nodup_pytorch_train_pytorch_transformer_ag8_specaug/decode_eval2000_decode/scoring/hyp.swbd.ctm.filt.sys:|   Sum/Avg     |   1831      21395    |   92.2        5.6        2.2        1.2         9.0       42.2   |
```

# transformer with BPE 2000
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/*.sys
exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/hyp.callhm.ctm.filt.sys:| SPKR        | # Snt  # Wrd  | Corr     Sub     Del     Ins     Err   S.Err  |
exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/hyp.callhm.ctm.filt.sys:| Sum/Avg     | 2628   21594  | 81.3    14.8     3.9     3.4    22.1    59.5  |
exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/hyp.ctm.filt.sys:    | SPKR       | # Snt # Wrd  | Corr    Sub    Del    Ins     Err  S.Err |
exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/hyp.ctm.filt.sys:    | Sum/Avg    | 4459  42989  | 85.3   11.4    3.3    2.5    17.1   55.3 |
exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/hyp.swbd.ctm.filt.sys: | SPKR       | # Snt  # Wrd  | Corr     Sub    Del     Ins     Err   S.Err  |
exp/train_nodup_pytorch_bpe2000/decode_eval2000_decode/scoring/hyp.swbd.ctm.filt.sys: | Sum/Avg    | 1831   21395  | 89.3     8.0    2.7     1.5    12.2    49.3  |
```

# transformer with BPE 500 (epoch 50)
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.*.sys
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.callhm.ctm.filt.sys:|      SPKR            |      # Snt           # Wrd      |      Corr              Sub              Del              Ins              Err            S.Err      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.callhm.ctm.filt.sys:|      Sum/Avg         |      2628            21594      |      78.0             16.5              5.5              3.4             25.4             64.0      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.ctm.filt.sys:|     SPKR            |     # Snt           # Wrd      |     Corr             Sub             Del              Ins             Err           S.Err      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.ctm.filt.sys:|     Sum/Avg         |     4459            42989      |     82.7            12.8             4.5              2.5            19.8            59.0      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.swbd.ctm.filt.sys:|      SPKR           |      # Snt           # Wrd      |      Corr             Sub              Del              Ins              Err            S.Err      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_50epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.swbd.ctm.filt.sys:|      Sum/Avg        |      1831            21395      |      87.5             9.1              3.4              1.6             14.1             51.9      |
```

# transformer with BPE 500 (epoch 30)
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_*_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.*.sys
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.callhm.ctm.filt.sys:|      SPKR            |      # Snt           # Wrd      |      Corr              Sub              Del              Ins              Err            S.Err      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.callhm.ctm.filt.sys:|      Sum/Avg         |      2628            21594      |      77.7             17.1              5.2              3.7             26.0             64.5      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.ctm.filt.sys:|     SPKR            |     # Snt           # Wrd      |     Corr             Sub             Del              Ins             Err           S.Err      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.ctm.filt.sys:|     Sum/Avg         |     4459            42989      |     82.8            13.0             4.2              2.8            20.0            59.4      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.swbd.ctm.filt.sys:|      SPKR           |      # Snt           # Wrd      |      Corr             Sub              Del              Ins              Err            S.Err      |
exp/train_nodup_pytorch_transformer_e12_BPE500_lr5_30epoch/decode_eval2000_decode_pytorch_transformer_ctcweight0.3/scoring/hyp.swbd.ctm.filt.sys:|      Sum/Avg        |      1831            21395      |      87.9             9.0              3.2              1.9             14.0             52.0      |
```

# Initial RNN results
## BLSTMP
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_a01_pt_blstmp_e6/*p0.1_len0.0-0.0/result.txt
exp/train_nodup_a01_pt_blstmp_e6/decode_eval2000_beam20_eacc.best_p0.1_len0.0-0.0/result.txt:|  SPKR       |  # Snt     # Wrd  |  Corr      Sub       Del      Ins      Err     S.Err  |
exp/train_nodup_a01_pt_blstmp_e6/decode_eval2000_beam20_eacc.best_p0.1_len0.0-0.0/result.txt:|  Sum/Avg    |  4458     181952  |  81.7      9.3       8.9     13.3     31.5      79.1  |
exp/train_nodup_a01_pt_blstmp_e6/decode_rt03_beam20_eacc.best_p0.1_len0.0-0.0/result.txt:|  SPKR         | # Snt    # Wrd  |  Corr     Sub      Del      Ins     Err    S.Err  |
exp/train_nodup_a01_pt_blstmp_e6/decode_rt03_beam20_eacc.best_p0.1_len0.0-0.0/result.txt:|  Sum/Avg      | 8422    321545  |  78.2    10.6     11.2     12.3    34.1     79.6  |
exp/train_nodup_a01_pt_blstmp_e6/decode_train_dev_beam20_eacc.best_p0.1_len0.0-0.0/result.txt:|  SPKR       |  # Snt     # Wrd  |  Corr       Sub      Del       Ins      Err     S.Err  |
exp/train_nodup_a01_pt_blstmp_e6/decode_train_dev_beam20_eacc.best_p0.1_len0.0-0.0/result.txt:|  Sum/Avg    |  3999     235886  |  87.4       6.4      6.2       5.0     17.6      65.4  |
```

## VGGBLSTMP
```
$ grep -e Avg -e SPKR -m 2 exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_*_decode_rnn_ctcweight0.3/scoring/hyp.*.sys
exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_eval2000_decode_rnn_ctcweight0.3/scoring/hyp.callhm.ctm.filt.sys:|   SPKR          |   # Snt      # Wrd    |   Corr          Sub         Del         Ins         Err       S.Err    |
exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_eval2000_decode_rnn_ctcweight0.3/scoring/hyp.callhm.ctm.filt.sys:|   Sum/Avg       |   2628       21594    |   75.8         19.3         4.9         4.2        28.5        64.6    |
exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_eval2000_decode_rnn_ctcweight0.3/scoring/hyp.ctm.filt.sys:|   SPKR         |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err       S.Err   |
exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_eval2000_decode_rnn_ctcweight0.3/scoring/hyp.ctm.filt.sys:|   Sum/Avg      |   4459       42989   |   81.2       14.8         4.0        3.2       22.0        60.1   |
exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_eval2000_decode_rnn_ctcweight0.3/scoring/hyp.swbd.ctm.filt.sys:|   SPKR         |   # Snt      # Wrd    |   Corr         Sub         Del         Ins         Err       S.Err    |
exp/train_nodup_pytorch_vggblstmp_e3_BPE500/decode_eval2000_decode_rnn_ctcweight0.3/scoring/hyp.swbd.ctm.filt.sys:|   Sum/Avg      |   1831       21395    |   86.6        10.3         3.1         2.2        15.6        53.5    |
```
