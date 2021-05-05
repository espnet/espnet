# v.: tacotron2

## Environments

- date: ``
- python version: ``
- espnet version: ``
- chainer version: ``
- pytorch version: ``
- Git hash: ``

## Model files

- model link: 
- training config file: ``
- decoding config file: ``
- cmvn file: ``
- e2e file: ``
- e2e JSON file: ``
- dict file: ``

## Samples



## Results

### CER (model.loss.best)

|dataset|Snt|Char|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
train_nodev_pytorch_train_pytorch_tacotron2/train_dev/result.txt:|100|1424|47.7|27.4|24.9|15.7|68|93|
train_nodev_pytorch_train_pytorch_tacotron2/test/result.txt:|130|1922|47.5|25.8|26.7|11.4|63.9|93.8|

### WER (model.loss.best)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
train_nodev_pytorch_train_pytorch_tacotron2/train_dev/result.wrd.txt:|100|591|19.6|40.3|40.1|3.9|84.3|93|
train_nodev_pytorch_train_pytorch_tacotron2/test/result.wrd.txt:|130|773|20.2|37.6|42.2|4.9|84.7|93.8|


# Ground truth

## Results

### CER

|dataset|Snt|Char|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
ground_truth/train_dev/result.txt:|100|1424|93.4|3.4|3.2|4.8|11.4|51|
ground_truth/test/result.txt:|130|1922|95.3|2|2.7|1.6|6.2|42.3|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
ground_truth/train_dev/result.wrd.txt:|100|591|82.6|11|6.4|1.5|19|53|
ground_truth/test/result.wrd.txt:|130|773|88.2|8.5|3.2|1|12.8|43.8|
