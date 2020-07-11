# Default transformer
  - Environments (obtained by `$ get_sys_info.sh`)
      - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
	  - python version: `Python 3.7.3`
	  - espnet version: `espnet 0.3.1`
	  - chainer version: `chainer 6.0.0`
	  - pytorch version: `pytorch 1.0.1.post2`
	  - Git hash: `2525193c2c25dea5683086ef1b69f45bd1e050af`
  - Model files link `https://drive.google.com/file/d/1vnm7DMZSLprgp4AVLKt5w7jPz4KFy1wQ/view?usp=sharing`
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_worn_simu_u400k_cleaned_trim_sp/cmvn.ark`
    - e2e file: `exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/model.loss.best.ep.10`
    - e2e JSON file: `exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/model.json`
  - Decoding results:
```
CER (or TER) result in exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/decode_dev_gss12.ep.10/result.txt                                                                                            
|  SPKR    |  # Snt    # Wrd   |  Corr      Sub       Del      Ins      Err    S.Err  |                                                                                                                            
|  Sum/Avg |  7437     280767  |  69.3     13.1      17.7      9.2     39.9     82.4  |                                                                                                                            
write a WER result in exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/decode_dev_gss12.ep.10/result.wrd.txt                                                                                                
|  SPKR     |  # Snt    # Wrd   |  Corr       Sub      Del       Ins       Err    S.Err   |                                                                                                                        
|  Sum/Avg  |  7437     58881   |  52.2      32.4     15.4       7.4      55.3     81.1   |
```
