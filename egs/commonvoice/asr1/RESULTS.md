# First results (default pytorch Transformer setting with BPE, 100 epochs, single GPU)
## Environments
- date: `Fri Aug  9 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.4.2`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.0`
- Git hash: `65d159c93e0770c121a4867b977bc19b2672a64c`
  - Commit date: `Fri Aug 9 22:51:19 2019 +0900`

## Model information (bpemode = bpe)
The number of BPE units (150) are tuned by the validation perplexity of the language model.
- Model files (archived to `model_bpe.tar.gz` by `$ pack_model.sh`)
- Model link: https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh
- training config file: `conf/tuning/train_pytorch_transformer.yaml `
- decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
- cmvn file: `data/valid_train/cmvn.ark`
- e2e file: `exp/valid_train_pytorch_bpe/results/model.last10.avg.best`
- e2e JSON file: `exp/valid_train_pytorch_bpe/results/model.json`
- lm file: `exp/train_rnnlm_pytorch_lm_bpe150/rnnlm.model.best`
- lm JSON file: `exp/train_rnnlm_pytorch_lm_bpe150/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_decode_lm|4076|99544|98.5|1.0|0.6|0.2|1.7|10.3|
|decode_valid_test_decode_lm|3995|97952|98.5|1.0|0.6|0.2|1.8|10.1|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_decode_lm|4076|38374|98.0|1.6|0.4|0.2|2.2|10.3|
|decode_valid_test_decode_lm|3995|37837|98.0|1.7|0.4|0.3|2.3|10.1|
```

## Model information (bpemode = unigram)
- Model files (archived to `model_unigram.tar.gz` by `$ pack_model.sh`)
- Model link: https://drive.google.com/open?id=1x_sG1MD4FU7RbvvMbR53EChq59tBJ_Bu
- training config file: `conf/tuning/train_pytorch_transformer.yaml `
- decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
- cmvn file: `data/valid_train/cmvn.ark`
- e2e file: `exp/valid_train_pytorch_unigram/results/model.last10.avg.best`
- e2e JSON file: `exp/valid_train_pytorch_unigram/results/model.json`
- lm file: `exp/train_rnnlm_pytorch_lm_unigram150/rnnlm.model.best`
- lm JSON file: `exp/train_rnnlm_pytorch_lm_unigram150/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_decode_lm|4076|100570|98.3|1.0|0.7|0.3|2.0|11.8|
|decode_valid_test_decode_lm|3995|99017|98.3|1.0|0.7|0.3|1.9|10.8|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_decode_lm|4076|38374|97.8|1.8|0.4|0.4|2.5|11.8|
|decode_valid_test_decode_lm|3995|37837|97.9|1.7|0.4|0.3|2.4|10.8|
```
