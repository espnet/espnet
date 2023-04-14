## pretrain_Train_pytorch_train_specaug

* Model files (archived to model.tar.gz by <code>$ pack_model.sh</code>)
  - download link: <code>https://drive.google.com/file/d/1YUePEjk2Utgznr7sP0x4KdKCcPjbMM7C/view?usp=sharing</code>
  - training config file: <code>conf/train.yaml</code>
  - decoding config file: <code>conf/decode.yaml</code>
  - preprocess config file: <code>conf/specaug.yaml</code>
  - lm config file: <code>conf/lm.yaml</code>
  - cmvn file: <code>data/pretrain_Train/cmvn.ark</code>
  - e2e file: <code>exp/pretrain_Train_pytorch_train_specaug/results/model.val5.avg.best</code>
  - e2e json file: <code>exp/pretrain_Train_pytorch_train_specaug/results/model.json</code>
  - lm file: <code>exp/pretrainedlm/rnnlm.model.best</code>
  - lm JSON file: <code>exp/pretrainedlm/model.json</code>
  - dict file: <code>data/lang_char/pretrain_Train_unigram5000_units.txt</code>


## Environments
- date: `Wed Feb 16 09:06:58 CET 2022`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.8`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `19aabb415657c05a45467f9d8bb612db4764f6a1`
  - Commit date: `Tue Oct 19 12:00:34 2021 +0200`


### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_Test_model.val5.avg.best_decode_|1243|12648|96.3|1.6|2.1|0.2|3.9|15.8|
|decode_Val_model.val5.avg.best_decode_|1082|14858|92.7|3.2|4.1|0.9|8.2|38.2|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_Test_model.val5.avg.best_decode_|1243|6660|96.2|2.1|1.7|0.4|4.2|15.7|
|decode_Val_model.val5.avg.best_decode_|1082|7866|91.6|4.7|3.7|1.0|9.4|38.2|
