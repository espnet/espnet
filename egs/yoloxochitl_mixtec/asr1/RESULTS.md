# RESULTS (100 epoch using single GPU)
## Environments
- date: `Thu Jun 25 23:13:00 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.2`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.1.0`

## Pre-trained Model
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1daXJp3mpvOKYYuEcgNbIDRyp16Q0gjFg/view?usp=sharing
  - training config file: `conf/train.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/train_mixtec_surface_reserve/cmvn.ark`
  - e2e file: `exp/train_mixtec_surface_reserve_pytorch_mixtec_surface_reserve/results/model.last10.avg.best`
  - e2e JSON file: `exp/train_mixtec_surface_reserve_pytorch_mixtec_surface_reserve/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_mixtec_surface_reserve_unigram150/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_mixtec_surface_reserve_unigram150/model.json`
  - dict file: `data/lang_char`


## train_mixtec_surface_reserve_pytorch_mixtec_surface_reserve
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_mixtec_surface_reserve_decode_mixtec_surface_reserve|10218|687420|89.6|6.0|4.5|2.7|13.2|87.8|
|decode_test_mixtec_surface_reserve_decode_mixtec_surface_reserve|10112|688918|89.7|5.9|4.4|2.7|13.0|87.9|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_mixtec_surface_reserve_decode_mixtec_surface_reserve|10218|165748|80.3|15.6|4.1|3.2|22.9|87.8|
|decode_test_mixtec_surface_reserve_decode_mixtec_surface_reserve|10112|166168|80.5|15.5|4.1|3.2|22.7|87.9|

