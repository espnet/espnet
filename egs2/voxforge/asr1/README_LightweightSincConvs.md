# Lightweight Sinc Convolutions
## About Lightweight Sinc Convolutions

Instead of using precomputed features, end-to-end speech recognition can also be done directly from raw audio using Lightweight Sinc Convolutions, as described in [https://arxiv.org/abs/2010.07597](https://arxiv.org/abs/2010.07597).

The process steps (in `espnet_model.py`) of the default frontend and the Sinc convolutions frontend can be compared as follows:

|                       |              Default | Lightweight Sinc Convolutions |
|-----------------------|---------------------:|------------------------------:|
| 1. Feature extraction |      F-bank features |                Sliding Window |
| 2. Data augmentation  |          SpecAugment |                   SpecAugment |
| 3. Normalization      |          Global-CMVN |                   Global-CMVN |
| 4. Pre-encoder        |                    - | Lightweight Sinc Convs Module |
| 5. Encoder/Decoder    | RNN/Transformer/etc. | RNN/Transformer/etc.          |


In the default frontend, a transformation to the spectrum and filtered to obtain features.  SpecAugment as data augmentation and global cepstral mean and variance normalization is applied.

This frontend for Lightweight Sinc Convolutions:

1. The sliding window partitions the raw audio stream into frames of e.g. 400 samples (25ms).
2. As data augmentation, SpecAugment is applied. Note that for Sinc convolutions, data augmentation is performed in time domain (vs. in spectral domain in the default frontend).
3. Global cepstral mean and variance normalization is applied.
4. In the pre-encoder: The normalized, raw audio data frames are then put into the Lightwight Sinc convolutions module that contains a Sinc input layer and depthwise convolutions as coupling layer.

In step 5., the extracted features are then given to the encoder module of your favourite end-to-end architecture.


### Usage

To use sinc convolutions in your model instead of the default F-bank frontend, add the following lines to your yaml configuration file:
```yaml
frontend: sliding_window
frontend_conf:
    hop_length: 200
preencoder: sinc
```
Note: If the datasets sample rate is not 16kHz, this parameter should be configured in the `preencoder` configuration instead of the frontend, i.e.:
```yaml
preencoder_conf:
    fs: 16000
```
Further description of the configuration options for the Sliding Window module and the Sinc module can be found in the ESPnet2 documentation for the corresponding modules.

### Visualization of Sinc filters

Learned filters that are stored in a model file (`*.pth`) can be visualized with `plot_sinc_filters.py`, for example:
```sh
cd egs2/voxforge/asr1/
expdir=./exp/path-to-your-experiment/
pyscripts/utils/plot_sinc_filters.py ${expdir}/valid.acc.best.pth ${expdir}/plot_sinc_filters
```

### Reference

To cite this work:
```
@article{ kurzinger2020lightweight,
	author = {K{\"u}rzinger, Ludwig and Lindae, Nicolas and Klewitz, Palle and Rigoll, Gerhard},
	title = {Lightweight End-to-End Speech Recognition from Raw Audio Data Using Sinc-Convolutions},
	journal = {Proc. Interspeech 2020},
	year = {2020},
	pages = {1659--1663},
}
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
