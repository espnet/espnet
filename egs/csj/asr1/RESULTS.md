# Lightweight/Dynamic convolution results
|                                                                            |   | # Snt | # Wrd |Corr|Sub|Del|Ins|Err|S.Err  |
| ------- | --- | --     | --     | -  | - | - | - | - | - |
|./exp/train_nodup_sp_pytorch_train_pytorch_SA-DC/decode_eval1_decode_lm/result.txt:|Sum/Avg|1272|43897|95.4|2.9|1.7|0.8|5.3|52.4|
|./exp/train_nodup_sp_pytorch_train_pytorch_SA-DC2D/decode_eval2_decode_lm/result.txt:|Sum/Avg|1292|43623|96.7|2.2|1.1|0.5|3.8|49.4|
|./exp/train_nodup_sp_pytorch_train_pytorch_SA-DC/decode_eval3_decode_lm/result.txt:|Sum/Avg|1385|28225|96.7|2.3|1.1|0.7|4.1|34.9|

# Transformer results
## Pytorch backend Transformer without any hyperparameter tuning
  - Model files (archived to transformer.v1.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_nodup_sp/cmvn.ark`
    - e2e file: `exp/train_nodup_sp_pytorch_train/results/model.acc.best`
    - e2e JSON file: `exp/train_nodup_sp_pytorch_train/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_nodup_sp_pytorch_train/decode_eval1_decode_lm/result.txt
     | SPKR     | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
     | Sum/Avg  | 1272   43897 | 95.1    3.1    1.7    0.8    5.7   53.5 |
exp/train_nodup_sp_pytorch_train/decode_eval2_decode_lm/result.txt
     | SPKR     | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
     | Sum/Avg  | 1292   43623 | 96.7    2.1    1.1    0.5    3.8   49.6 |
exp/train_nodup_sp_pytorch_train/decode_eval3_decode_lm/result.txt
     | SPKR     | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
     | Sum/Avg  | 1385   28225 | 96.6    2.3    1.1    0.8    4.2   35.9 |
```

# RNN results
## Deep VGGBLSTM with pytorch backend + dropout + Speed perturbation + CTC joint decoding + LM rescoreing
  - Model files (archived to transformer.v1.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1ALvD4nHan9VDJlYJwNurVr7H7OV0j2X9
    - training config file: `conf/tuning/train_rnn.yaml`
    - decoding config file: `conf/tuning/decode_rnn.yaml`
    - cmvn file: `data/train_nodup_sp/cmvn.ark`
    - e2e file: `exp/train_nodup_sp_pytorch_train_rnn/results/model.acc.best`
    - e2e JSON file: `exp/train_nodup_sp_pytorch_train_rnn/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_nodup_sp_pytorch_train_rnn/decode_eval1_decode_rnn_lm/result.txt
   | SPKR     | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
   | Sum/Avg  | 1272    43897 | 94.4     3.7    1.9     0.9    6.5    55.9 |
exp/train_nodup_sp_pytorch_train_rnn/decode_eval2_decode_rnn_lm/result.txt
   | SPKR     | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
   | Sum/Avg  | 1292    43623 | 96.0     2.8    1.2     0.6    4.6    54.5 |
exp/train_nodup_sp_pytorch_train_rnn/decode_eval3_decode_rnn_lm/result.txt
   | SPKR     | # Snt   # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
   | Sum/Avg  | 1385    28225 | 95.8     2.9    1.3     0.9    5.1    37.9 |
```

## Deep VGGBLSTM with pytorch backend + Dropout + Speed perturbation + CTC joint decoding + LM rescoring
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_sp_pytorch_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_drop0.2_bs24_mli800_mlo150/decode_eval1_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3_2layer_unit650_sgd_bs256/|1272|43897|94.3|3.8|1.9|0.9|6.6|56.8|
|exp/train_nodup_sp_pytorch_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_drop0.2_bs24_mli800_mlo150/decode_eval2_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3_2layer_unit650_sgd_bs256/|1292|43623|95.9|2.9|1.2|0.6|4.8|55.7|
|exp/train_nodup_sp_pytorch_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_drop0.2_bs24_mli800_mlo150/decode_eval3_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3_2layer_unit650_sgd_bs256/|1385|28225|95.8|3.0|1.2|0.9|5.0|38.0|

## Deep VGGBLSTM with pytorch backend + Dropout (Early stoping 11 epoch) + CTC joint decoding + LM rescoring
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_pytorch_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_drop0.2_bs24_mli800_mlo150/decode_eval1_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3_2layer_unit650_sgd_bs256/|1272|43897|93.9|4.2|2.0|1.0|7.1|58.0|
|exp/train_nodup_pytorch_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_drop0.2_bs24_mli800_mlo150/decode_eval2_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3_2layer_unit650_sgd_bs256/|1292|43623|95.7|3.1|1.2|0.7|5.0|55.7|
|exp/train_nodup_pytorch_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim1024_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_drop0.2_bs24_mli800_mlo150/decode_eval3_beam30_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3_2layer_unit650_sgd_bs256/|1385|28225|95.2|3.4|1.4|1.1|5.9|43.2|


## Deep VGGBLSTM (elayers=4) with chainer backend + CTC joint decoding + LM rescoreing + Subword(nbpe=5000)
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim320_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150_unigram5000/decode_eval1_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1272|43897|93.2|4.5|2.3|1.1|8.0|61.4|
|exp/train_nodup_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim320_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150_unigram5000/decode_eval2_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1292|43623|94.8|3.6|1.6|0.9|6.1|60.0|
|exp/train_nodup_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj1024_d1_unit1024_location_adim320_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150_unigram5000/decode_eval3_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1385|28225|95.0|3.5|1.5|1.1|6.1|43.0|

## Deep VGGBLSTM (elayers=4) with chainer backend + CTC joint decoding + LM rescoreing
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj320_d1_unit1024_location_adim320_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1272|43897|93.7|4.3|2.0|1.0|7.3|59.6|
|exp/train_nodup_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj320_d1_unit1024_location_adim320_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1292|43623|95.4|3.4|1.2|0.8|5.3|58.7|
|exp/train_nodup_vggblstm_e4_subsample1_2_2_1_1_unit1024_proj320_d1_unit1024_location_adim320_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs32_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1385|28225|95.2|3.5|1.3|1.1|5.9|41.8|

## Deep VGGBLSTMP (elayers=6) with chainer backend + CTC joint decoding + LM rescoreing
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1272|43897|92.6|5.3|2.1|1.3|8.7|63.1|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1292|43623|94.7|4.0|1.3|0.9|6.2|60.5|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.3/|1385|28225|94.3|4.2|1.5|1.2|6.9|45.5|


## Deep VGGBLSTMP (elayers=6) with chainer backend + CTC joint decoding
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.0/|1272|43897|91.6|6.0|2.3|1.4|9.7|66.5|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.0/|1292|43623|94.1|4.6|1.3|1.0|6.9|64.5|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.3_rnnlm0.0/|1385|28225|93.9|4.7|1.4|1.4|7.5|47.7|


## Deep VGGBLSTMP (elayers=6) with chainer backend
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.1_len0.1-0.5/|1272|43897|91.4|6.4|2.3|1.6|10.2|67.6|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.1_len0.1-0.5/|1292|43623|93.7|5.1|1.3|1.2|7.5|65.2|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.1_len0.1-0.5/|1385|28225|93.6|5.0|1.4|1.6|8.0|47.9|


## Deep BLSTMP (elayers=6) with pytorch backend
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.1_len0.1-0.5/|1272|43897|90.9|6.7|2.5|1.6|10.7|68.6|
|exp/train_nodup_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.1_len0.1-0.5/|1292|43623|93.2|5.3|1.5|1.2|8.0|66.6|
|exp/train_nodup_blstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.1_len0.1-0.5/|1385|28225|92.8|5.6|1.6|1.6|8.8|51.0|


## BLMSTP with pytorch backend
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.1_len0.1-0.5/|1272|43897|89.6|7.6|2.8|1.8|12.1|70.6|
|exp/train_nodup_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.1_len0.1-0.5/|1292|43623|92.1|6.2|1.7|1.3|9.2|69.5|
|exp/train_nodup_blstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.1_len0.1-0.5/|1385|28225|91.1|6.8|2.1|2.0|10.9|56.2|


## Deep VGGBLSTMP (elayers=6)
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.1_len0.1-0.5/|1272|43897|91.4|6.4|2.3|1.6|10.2|67.6|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.1_len0.1-0.5/|1292|43623|93.7|5.1|1.3|1.2|7.5|65.2|
|exp/train_nodup_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.1_len0.1-0.5/|1385|28225|93.6|5.0|1.4|1.6|8.0|47.9|

## VGGBLSTMP, adadelta with eps decay monitoring validation acc, maxlenraito=0.5, minlenratio=0.1, and penalty=0.1
|dataset| Snt | Wrd| Corr | Sub | Del | Ins | Err | S.Err|
|---|---|---|---|---|---|---|---|---|
|exp/train_nodup_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval1_beam20_eacc.best_p0.1_len0.1-0.5/|1272|43897|90.3|7.0|2.6|1.6|11.3|70.5|
|exp/train_nodup_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval2_beam20_eacc.best_p0.1_len0.1-0.5/|1292|43623|93.3|5.3|1.4|1.1|7.8|66.1|
|exp/train_nodup_vggblstmp_e4_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_bs30_mli800_mlo150/decode_eval3_beam20_eacc.best_p0.1_len0.1-0.5/|1385|28225|92.6|6.0|1.5|1.8|9.2|53.9|
