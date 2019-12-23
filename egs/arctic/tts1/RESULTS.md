# v.0.5.3 / Initial results of fine-tuning

- Initial model: `mailabs.en_US.judy.transformer.v1.single`
- Target speaker: `slt`
- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Fri Oct 11 10:02:25 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `23ce84d1d2d8ac1e6ea2839035a7ff3dc976658b`
  - Commit date: `Fri Oct 11 10:00:46 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1f4udbcfsuWw0OkpjpB4ohgggkz2IMojq
- training config file: `conf/train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `downloads/mailabs.en_US.judy.transformer.v1.single/data/en_US_judy_train_trim/cmvn.ark`
- e2e file: `exp/slt_train_no_dev_pytorch_train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/slt_train_no_dev_pytorch_train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single/results/model.json`
- dict file: `downloads/mailabs.en_US.judy.transformer.v1.single/data/lang_1char/en_US_judy_train_trim_units.txt`

## Samples

https://drive.google.com/open?id=1oF8rsQrJccZEJ5mdLFyhiGVQR3q503fN

## Results

### CER

|dataset|Snt|Char|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single/dev/result.txt:|100|4008|97.6|1.1|1.3|0.6|3.0|32.0|
train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single/eval/result.txt:|100|4001|96.9|1.4|1.7|0.7|3.8|37.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single/dev/result.wrd.txt:|100|929|94.8|4.2|1.0|0.4|5.6|32.0|
train_pytorch_transformer.v1.single.finetune.mailabs.en_US.judy.transformer.v1.single/eval/result.wrd.txt:|100|878|93.5|5.5|1.0|1.1|7.6|39.0|

# Ground truth

## Results

### CER

|dataset|Snt|Char|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
ground_truth/dev/result.txt:|100|4008|99.7|0.1|0.2|0.3|0.6|10.0|
ground_truth/eval/result.txt:|100|4001|99.4|0.4|0.2|0.2|0.8|15.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
ground_truth/dev/result.wrd.txt:|100|929|98.6|1.3|0.1|0.1|1.5|10.0|
ground_truth/eval/result.wrd.txt:|100|878|97.4|2.3|0.3|0.7|3.3|18.0|
