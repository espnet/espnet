# Lightweight Sinc Convolutions
## About Lightweight Sinc Convolutions


Instead of using precomputed features, end-to-end speech recognition can also be done directly from raw audio using sinc convolutions, as described in [https://arxiv.org/abs/2010.07597](https://arxiv.org/abs/2010.07597).
To use sinc convolutions in your model instead of the default f-bank frontend, add the following lines to your yaml configuration file:

```yaml
frontend: sliding_window
frontend_conf:
    hop_length: 200
preencoder: sinc
```

Note that this method also performs data augmentation in time domain (vs. in spectral domain in the default frontend).
Learned filters that are stored in a model file (`*.pth`) can be plotted with `plot_sinc_filters.py`, for example:
```sh
cd egs2/voxforge/asr1/
expdir=./exp/path-to-your-experiment/
pyscripts/utils/plot_sinc_filters.py ${expdir}/valid.acc.best.pth ${expdir}/plot_sinc_filters
```


## [Sinc-BLSTMP with hop_size=240](conf/tuning/train_asr_sinc_rnn.yaml)
### Environments
- date: `Wed Nov 25 16:37:11 CET 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.4.0`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|13235|59.0|35.6|5.4|6.0|47.1|97.6|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|12990|57.9|35.7|6.3|5.6|47.7|98.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|79133|87.8|6.7|5.5|2.7|14.9|97.6|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|77966|86.9|7.0|6.2|2.5|15.6|98.2|



## [Sinc-Transformer with hop_size=200](conf/tuning/train_asr_sinc_transformer.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|13235|57.4|36.1|6.6|4.3|47.0|97.9|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|12990|55.6|37.1|7.3|4.5|48.9|98.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|79133|87.5|6.4|6.1|2.4|14.8|97.9|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|77966|87.1|6.6|6.3|2.4|15.3|98.3|

