# Default transformer
  - Environments (obtained by `$ get_sys_info.sh`)
      - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
	  - python version: `Python 3.7.3`
	  - espnet version: `espnet 0.3.1`
	  - chainer version: `chainer 6.0.0`
	  - pytorch version: `pytorch 1.0.1.post2`
	  - Git hash: `2525193c2c25dea5683086ef1b69f45bd1e050af`
  - Model files (archived to archive.tgz by `$ pack_model.sh`)
      - model link: https://drive.google.com/open?id=1pAQV3fjJu0AlD1JiBsDCgRofegeiLFbA
	  - training config file: `conf/train.yaml`
	  - decoding conf-if file: `conf/decode.yaml`
	  - cmvn file: `data/train_worn_u200k/cmvn.ark`
	  - e2e file: `exp/train_worn_u200k_pytorch_train/results/model.last10.avg.best`
	  - e2e JSON file: `exp/train_worn_u200k_pytorch_train/results/model.json`
	  - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
	  - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
	  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_worn_u200k_pytorch_train/decode_dev_worn_decode_lm/result.wrd.txt
| SPKR    | # Snt   # Wrd | Corr     Sub     Del     Ins    Err   S.Err  |
| Sum/Avg | 7437    58881 | 49.6    38.3    12.1     9.8   60.2    77.1  |
exp/train_worn_u200k_pytorch_train/decode_dev_beamformit_ref_decode_lm/result.wrd.txt
|  SPKR    |  # Snt    # Wrd  |  Corr     Sub      Del      Ins      Err    S.Err  |
|  Sum/Avg |  7437     58881  |  31.0    49.7     19.2     18.1     87.1     89.7  |
```

# Initial trial with CTC and LM weight tuning

- Results

- ```bash
  $ grep -e Avg -e SPKR -m 2 exp/train_worn_u200k_ch_vggblstmp_mtlalpha0.1/decode_dev_beamformit_ref_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.1_rnnlm0.1/result.wrd.txt
  |     SPKR       |    # Snt          # Wrd     |    Corr            Sub           Del            Ins           Err          S.Err     |
  |     Sum/Avg    |    7437           58881     |    28.2           46.7          25.1           16.3          88.1           85.3     |
  $ grep -e Avg -e SPKR -m 2 exp/train_worn_u200k_ch_vggblstmp_mtlalpha0.1/decode_dev_worn_beam20_eacc.best_p0.0_len0.0-0.0_ctcw0.1_rnnlm0.1/result.wrd.txt
  |    SPKR      |    # Snt         # Wrd    |    Corr           Sub          Del           Ins          Err         S.Err    |
  |    Sum/Avg   |    7437          58881    |    50.2          38.2         11.5           9.6         59.3          73.6    |
  ```

  
