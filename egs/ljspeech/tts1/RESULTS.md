# LJSpeech E2E-TTS results and samples

## v.0.3.0: 1024 pt window / 256 pt shift + default taco2 + GL 1000 iters

- Samples: https://drive.google.com/open?id=1NclM7_WaoL_Joy71e1bAUfsn_Hcy6HZD

## v.0.4.0: tacotron2.v1 1024 pt window / 256 pt shift / GL 1000 iters

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Mon Jun 10 10:15:51 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `c86e9311641f59fa397912d0bd2b9c0c599a1127`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1zVMwNZ0-VMILm83UEQ2oUqWI89y39Y_b
  - training config file: `conf/tuning/train_pytorch_tacotron2.v1.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.json`

- Samples: https://drive.google.com/open?id=1ZIDPpb1Bt9V8mrnJCCptMcpIH3SpuyrD

## v.0.4.0: tacotron2.v2 1024 pt window / 256 pt shift / GL 1000 iters

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Fri Jun 14 10:51:01 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `c86e9311641f59fa397912d0bd2b9c0c599a1127`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1qesE6-WVDdwZe_l5rvbXTtZQcQbJ53tg
  - training config file: `conf/tuning/train_pytorch_tacotron2.v2.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.json`

- Samples: https://drive.google.com/open?id=1cKPDQjLGs7OD8xopSK3YWIGGth37GRSm
