## RNN with old config
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
    - training config file: `conf/train.yaml`    
    - decoding config file: `conf/decode.yaml`  
    - e2e file: `exp/ihm_train_pytorch_train/results/model.loss.best`    
    - e2e JSON file: `exp/ihm_train_pytorch_train/results/model.json`    
    - lm file: `exp/train_rnnlm_pytorch_lm_word20000/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_word20000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/ihm_train_pytorch_train/decode_ihm_dev_decode_lm_word20000/result.wrd.txt
| SPKR                         | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                      |13059  94914 | 61.6   31.1    7.3    7.3   45.7   73.5 |
exp/ihm_train_pytorch_train/decode_ihm_eval_decode_lm_word20000/result.wrd.txt
| SPKR                         | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                      |12612  89635 | 57.6   36.4    6.0    8.3   50.7   71.1 |
```

## RNN with shallow wide encoder
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
    - training config file: `conf/train_rnn.yaml`
    - decoding config file: `conf/decode_rnn.yaml`
    - e2e file: `exp/ihm_train_pytorch_train_rnn/results/model.acc.best`
    - e2e JSON file: `exp/ihm_train_pytorch_train_rnn/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_word20000/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_word20000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/ihm_train_pytorch_train_rnn/decode_ihm_dev_decode_rnn_lm_word20000/result.wrd.txt
| SPKR                         | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                      |13059  94914 | 73.8   19.0    7.2    4.2   30.4   66.2 |
exp/ihm_train_pytorch_train_rnn/decode_ihm_eval_decode_rnn_lm_word20000/result.wrd.txt
| SPKR                         | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg                      |12612  89635 | 72.3   21.7    6.1    4.2   32.0   64.3 |
```

## Transformer with large encoder (epoch 100)
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
   - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
   - training config file: `conf/train_transformer.yaml`
   - decoding config file: `conf/decode_transformer.yaml`
   - e2e file: `exp/ihm_train_pytorch_train_transformer/results/model.acc.best`
   - e2e JSON file: `exp/ihm_train_pytorch_train_transformer/results/model.json`
   - lm file: `exp/train_rnnlm_pytorch_lm_word20000/rnnlm.model.best`
   - lm JSON file: `exp/train_rnnlm_pytorch_lm_word20000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/ihm_train_pytorch_train_transformer/decode_ihm_dev_decode_transformer_lm_word20000/result.wrd.txt
| SPKR                          |  # Snt  # Wrd  | Corr      Sub     Del     Ins      Err   S.Err  |
| Sum/Avg                       | 13059   94914  | 76.6     16.7     6.7     3.2     26.6    63.0  |
exp/ihm_train_pytorch_train_transformer/decode_ihm_eval_decode_transformer_lm_word20000/result.wrd.txt
| SPKR                          |  # Snt  # Wrd  |  Corr     Sub      Del     Ins      Err   S.Err  |
| Sum/Avg                       | 12612   89635  |  75.0    18.4      6.6     2.9     27.9    60.9  |
```
## Transformer with large encoder (epoch 200)
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
   - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
   - training config file: `conf/train_transformer.yaml`
   - decoding config file: `conf/decode_transformer.yaml`
   - e2e file: `exp/ihm_train_pytorch_train_transformer/results/model.acc.best`
   - e2e JSON file: `exp/ihm_train_pytorch_train_transformer/results/model.json`
   - lm file: `exp/train_rnnlm_pytorch_lm_word20000/rnnlm.model.best`
   - lm JSON file: `exp/train_rnnlm_pytorch_lm_word20000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/ihm_train_pytorch_train_transformer_200/decode_ihm_dev_decode_transformer_lm_word20000/result.wrd.txt
|  SPKR                          |  # Snt   # Wrd  |  Corr     Sub      Del      Ins      Err    S.Err  |
|  Sum/Avg                       | 13059    94914  |  77.0    16.9      6.2      3.3     26.3     62.9  |
exp/ihm_train_pytorch_train_transformer_200/decode_ihm_eval_decode_transformer_lm_word20000/result.wrd.txt
|  SPKR                          |  # Snt   # Wrd  |  Corr      Sub      Del      Ins      Err    S.Err  |
|  Sum/Avg                       | 12612    89635  |  75.4     18.4      6.2      3.1     27.7     60.6  |
```
## Transformer with large encoder with speed perturbation based data augmentation.(epoch 100)
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)  
    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)
    - training config file: `conf/train_transformer.yaml`    
    - decoding config file: `conf/decode_transformer.yaml`    
    - e2e file: `exp/ihm_train_pytorch_train/results/model.acc.best`    
    - e2e JSON file: `exp/ihm_train_pytorch_train/results/model.json`    
    - lm file: `exp/train_rnnlm_pytorch_lm_word20000/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_word20000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/ihm_train_pytorch_augment/decode_ihm_dev_decode_transformer_lm_word20000/result.wrd.txt
| SPKR                         | # Snt # Wrd  | Corr    Sub    Del    Ins     Err  S.Err |
| Sum/Avg                      |13059  94914  | 78.0   15.9    6.0    3.0    25.0   61.5 |
exp/ihm_train_pytorch_augment/decode_ihm_eval_decode_transformer_lm_word20000/result.wrd.txt
| SPKR                         | # Snt  # Wrd | Corr    Sub     Del    Ins    Err   S.Err |
| Sum/Avg                      |12612   89635 | 76.8   16.9     6.3    2.4   25.6    59.3 |
```
