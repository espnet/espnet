# v.0.5.3 / Initial results of fine-tuning

- Initial model: `jsut.24k.phn.transformer`
- Target speaker: `jvs010`
- FFT in points: 2048
- Shift in points: 300
- Window length in points: 1200
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Sat Oct 12 00:05:32 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `dd4c06364b588b42d7906304570ede9088d0cc29`
  - Commit date: `Sat Oct 12 00:02:45 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1pWbfev3cC6mJSuz7liRqvdYguLYhKWIn
- training config file: `conf/train_pytorch_transformer.finetune.jsut.24k.phn.transformer.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `downloads/jsut.24k.phn.transformer/data/train_no_dev/cmvn.ark`
- e2e file: `exp/jvs010_phn_train_no_dev_pytorch_train_pytorch_transformer.finetune.jsut.24k.phn.transformer/results/model.loss.best`
- e2e JSON file: `exp/jvs010_phn_train_no_dev_pytorch_train_pytorch_transformer.finetune.jsut.24k.phn.transformer/results/model.json`
- dict file: `downloads/jsut.24k.phn.transformer/data/lang_1phn/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1jvASMlDrQ8s2mQTRqL-Ru5sELMMYaGNp
