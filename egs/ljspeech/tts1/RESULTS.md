# LJSpeech E2E-TTS results and samples

## v.0.3.0: tacotron2 1024 pt window / 256 pt shift + default taco2 + GL 1000 iters

- Samples: https://drive.google.com/open?id=1NclM7_WaoL_Joy71e1bAUfsn_Hcy6HZD

## v.0.4.0: tacotron2.v1 1024 pt window / 256 pt shift / GL 1000 iters / R=2 / location-sensitive

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Mon Jun 10 10:15:51 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `c86e9311641f59fa397912d0bd2b9c0c599a1127`

- Model files (archived to train_no_dev_pytorch_train_pytorch_tacotron2.v1.tar.gz.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1dKzdaDpOkpx7kWZnvrvx2De7eZEdPHZs
  - training config file: `conf/tuning/train_pytorch_tacotron2.v1.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.json`
  - dict file: `data/lang_1char/train_no_dev_units.txt`

- Samples: https://drive.google.com/open?id=1ZIDPpb1Bt9V8mrnJCCptMcpIH3SpuyrD

## v.0.4.0: tacotron2.v2 1024 pt window / 256 pt shift / GL 1000 iters/ R=1 / forward with transition agent

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Fri Jun 14 10:51:01 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `c86e9311641f59fa397912d0bd2b9c0c599a1127`

- Model files (archived to train_no_dev_pytorch_train_pytorch_tacotron2.v2.tar.gz.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=11T9qw8rJlYzUdXvFjkjQjYrp3iGfQ15h
  - training config file: `conf/tuning/train_pytorch_tacotron2.v2.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.json`
  - dict file: `data/lang_1char/train_no_dev_units.txt`

- Samples: https://drive.google.com/open?id=1cKPDQjLGs7OD8xopSK3YWIGGth37GRSm

## v.0.4.0: tacotron2.v3 1024 pt window / 256 pt shift / GL 1000 iters / R=1 / location-sensitive / guided-attention

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Sun Jun 16 10:03:47 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `267da3161cefeae72e9a44bd15e74c0d18591fb6`

- Model files (archived to train_no_dev_pytorch_train_pytorch_tacotron2.v3.tar.gz.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1hiZn14ITUDM1nkn-GkaN_M3oaTOUcn1n
  - training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.json`
  - dict file: `data/lang_1char/train_no_dev_units.txt`
- Samples: https://drive.google.com/open?id=18JgsOCWiP_JkhONasTplnHS7yaF_konr

## v.0.4.0: transofrmer.v1 1024 pt window / 256 pt shift / GL 1000 iters / R=1 / Large

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Sun Jun 16 10:03:47 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `267da3161cefeae72e9a44bd15e74c0d18591fb6`

- Model files (archived to train_no_dev_pytorch_train_pytorch_transformer.v1.tar.gz.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS
  - training config file: `conf/tuning/train_pytorch_transformer.v1.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v1/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v1/results/model.json`
  - dict file: `data/lang_1char/train_no_dev_units.txt`

- Samples: https://drive.google.com/open?id=14EboYVsMVcAq__dFP1p6lyoZtdobIL1X

## v.0.4.0: transofrmer.v2 1024 pt window / 256 pt shift / GL 1000 iters / R=3 / Small

- Environments (obtained by `$ get_sys_info.sh`)
  - date: `Sun Jun 16 10:03:47 JST 2019`
  - system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
  - python version: `Python 3.7.3`
  - espnet version: `espnet 0.3.1`
  - chainer version: `chainer 5.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `267da3161cefeae72e9a44bd15e74c0d18591fb6`

- Model files (archived to train_no_dev_pytorch_train_pytorch_transformer.v2.tar.gz.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1xxAwPuUph23RnlC5gym7qDM02ZCW9Unp
  - training config file: `conf/tuning/train_pytorch_transformer.v2.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_no_dev/cmvn.ark`
  - e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v2/results/model.last1.avg.best`
  - e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v2/results/model.json`
  - dict file: `data/lang_1char/train_no_dev_units.txt`

- Samples: https://drive.google.com/open?id=1TqY5cvA2azhl7Xi3E1LFRpsTajlHxO_P
