## pretrain_Train_pytorch_train_specaug

* Model files (archived to model.tar.gz by <code>$ pack_model.sh</code>)
  - download link: <code>https://drive.google.com/file/d/1YUePEjk2Utgznr7sP0x4KdKCcPjbMM7C/view?usp=sharing</code>
  - training config file: <code>conf/train.yaml</code>
  - decoding config file: <code>conf/decode.yaml</code>
  - preprocess config file: <code>conf/specaug.yaml</code>
  - lm config file: <code>conf/lm.yaml</code> 
  - cmvn file: <code>data/train_960/cmvn.ark</code> and <code>data/pretrain_Train/cmvn.ark</code>
  - e2e file: <code>exp/pretrain_Train_pytorch_train_specaug/results/model.val5.avg.best</code>
  - e2e json file: <code>exp/pretrain_Train_pytorch_train_specaug/results/model.json</code>
  - lm file: <code>exp/pretrainedlm/rnnlm.model.best</code>
  - lm JSON file: <code>exp/pretrainedlm/model.json</code>
  - dict file: <code>data/lang_char/pretrain_Train_unigram5000_units.txt</code>

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
