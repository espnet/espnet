<div align="left"><img src="doc/image/espnet_logo1.png" width="550"/></div>

# ESPnet: end-to-end speech processing toolkit

|system/pytorch ver.|1.0.1|1.1.0|1.2.0|1.3.1|1.4.0|1.5.0|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ubuntu18/python3.8/pip||||||[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|
|ubuntu18/python3.7/pip|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|
|ubuntu18/python3.6/conda||||||[![CircleCI](https://circleci.com/gh/espnet/espnet.svg?style=svg)](https://circleci.com/gh/espnet/espnet)|
|ubuntu16/python3.6/conda||||||[![CircleCI](https://circleci.com/gh/espnet/espnet.svg?style=svg)](https://circleci.com/gh/espnet/espnet)|
|debian9/python3.6/conda||||||[![CircleCI](https://circleci.com/gh/espnet/espnet.svg?style=svg)](https://circleci.com/gh/espnet/espnet)|
|centos7/python3.6/conda||||||[![CircleCI](https://circleci.com/gh/espnet/espnet.svg?style=svg)](https://circleci.com/gh/espnet/espnet)|
|[docs/coverage] python3.8||||||[![Build Status](https://travis-ci.org/espnet/espnet.svg?branch=master)](https://travis-ci.org/espnet/espnet)|

[![PyPI version](https://badge.fury.io/py/espnet.svg)](https://badge.fury.io/py/espnet)
[![Python Versions](https://img.shields.io/pypi/pyversions/espnet.svg)](https://pypi.org/project/espnet/)
[![Downloads](https://pepy.tech/badge/espnet)](https://pepy.tech/project/espnet)
[![codecov](https://codecov.io/gh/espnet/espnet/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Mergify Status](https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/espnet/espnet&style=flat)](https://mergify.io)
[![Gitter](https://badges.gitter.im/espnet-en/community.svg)](https://gitter.im/espnet-en/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[**Docs**](https://espnet.github.io/espnet/)
| [**Example**](https://github.com/espnet/espnet/tree/master/egs)
| [**Docker**](https://github.com/espnet/espnet/tree/master/docker)
| [**Notebook**](https://github.com/espnet/notebook)
| [**Tutorial (2019)**](https://github.com/espnet/interspeech2019-tutorial)

[**Master**](https://github.com/espnet/espnet/tree/master)
| [**Develop**](https://github.com/espnet/espnet/tree/develop)
| [**Release**](https://github.com/espnet/espnet/releases)

ESPnet is an end-to-end speech processing toolkit, mainly focuses on end-to-end speech recognition and end-to-end text-to-speech.
ESPnet uses [chainer](https://chainer.org/) and [pytorch](http://pytorch.org/) as a main deep learning engine,
and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for speech recognition and other speech processing experiments.

## Key Features

### Kaldi style complete recipe
- Support numbers of `ASR` recipes (WSJ, Switchboard, CHiME-4/5, Librispeech, TED, CSJ, AMI, HKUST, Voxforge, REVERB, etc.)
- Support numbers of `TTS` recipes with a similar manner to the ASR recipe (LJSpeech, LibriTTS, M-AILABS, etc.)
- Support numbers of `ST` recipes (Fisher-CallHome Spanish, Libri-trans, IWSLT'18, How2, Must-C, Mboshi-French, etc.)
- Support numbers of `MT` recipes (IWSLT'16, the above ST recipes etc.)
- Support speech separation and recognition recipe (WSJ-2mix)
- Support voice conversion recipe (VCC2020 baseline) (new!)


### ASR: Automatic Speech Recognition

- **State-of-the-art performance** in several ASR benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- **Hybrid CTC/attention** based end-to-end ASR
  - Fast/accurate training with CTC/attention multitask training
  - CTC/attention joint decoding to boost monotonic alignment decoding
  - Encoder: VGG-like CNN + BiRNN (LSTM/GRU), sub-sampling BiRNN (LSTM/GRU) or Transformer
- Attention: Dot product, location-aware attention, variants of multihead
- Incorporate RNNLM/LSTMLM/TransformerLM/N-gram trained only with text data
- Batch GPU decoding
- **Transducer** based end-to-end ASR
  - Available: RNN-Transducer, Transformer-Transducer, mixed Transformer/RNN-Transducer
  - Also support: attention mechanism (RNN-decoder), pre-init w/ LM (RNN-decoder), VGG-Transformer (encoder)

### TTS: Text-to-speech
- Tacotron2 based end-to-end TTS
- Transformer based end-to-end TTS
- Feed-forward Transformer (a.k.a. FastSpeech) based end-to-end TTS (new!)

### ST: Speech Translation & MT: Machine Translation
- **State-of-the-art performance** in several ST benchmarks (comparable/superior to cascaded ASR and MT)
- Transformer based end-to-end ST (new!)
- Transformer based end-to-end MT (new!)

### VC: Voice conversion
- End-to-end VC based on cascaded ASR+TTS (new!)
- Baseline system for Voice Conversion Challenge 2020!

### DNN Framework
- Flexible network architecture thanks to chainer and pytorch
- Flexible front-end processing thanks to [kaldiio](https://github.com/nttcslab-sp/kaldiio) and HDF5 support
- Tensorboard based monitoring

## Installation
- If you intend to do full experiments including DNN training, then see [Installation](https://espnet.github.io/espnet/installation.html).
- If you just need the Python module only: 
    ```bash
    pip install torch  # Install some dependencies manually
    pip install espnet
    # To install latest
    # pip install git+https://github.com/espnet/espnet
    ```

## Usage
See [Usage](https://espnet.github.io/espnet/tutorial.html).

## Docker Container

go to [docker/](docker/) and follow [instructions](https://espnet.github.io/espnet/docker.html).

## About ESPnet2
See [ESPnet2](https://espnet.github.io/espnet/espnet2_tutorial.html).

## Contribution
Thank you for taking times for ESPnet! Any contributions to ESPNet are welcome and feel free to ask any questions or requests to [issues](https://github.com/espnet/espnet/issues).
If it's the first contribution to ESPnet for you,  please follow the [contribution guide](CONTRIBUTING.md).

### Branching strategy

- [master](https://github.com/espnet/espnet/tree/master): Hot fix, adding new recipes, fix typo, update README.md
- [develop](https://github.com/espnet/espnet/tree/develop): Adding new feature, refactoring, adding test codes
- [release](https://github.com/espnet/espnet/releases): The specific commit by tag

## Results and demo

You can find useful tutorials and demos in [Interspeech 2019 Tutorial](https://github.com/espnet/interspeech2019-tutorial)

### ASR results

<details><summary>expand</summary><div>


We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

| Task                   | CER (%) | WER (%) | Pretrained model                                                                                                                                                      |
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| Aishell dev            | 6.0     | N/A     | [link](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#transformer-result-default-transformer-with-initial-learning-rate--10-and-epochs--50) |
| Aishell test           | 6.6     | N/A     | same as above                                                                                                                                                         |
| Common Voice dev       | 1.7     | 2.2     | [link](https://github.com/espnet/espnet/blob/master/egs/commonvoice/asr1/RESULTS.md#first-results-default-pytorch-transformer-setting-with-bpe-100-epochs-single-gpu) |
| Common Voice test      | 1.8     | 2.3     | same as above                                                                                                                                                         |
| CSJ eval1              | 5.7     | N/A     | [link](https://github.com/espnet/espnet/blob/master/egs/csj/asr1/RESULTS.md#pytorch-backend-transformer-without-any-hyperparameter-tuning)                            |
| CSJ eval2              | 3.8     | N/A     | same as above                                                                                                                                                         |
| CSJ eval3              | 4.2     | N/A     | same as above                                                                                                                                                         |
| HKUST dev              | 23.5    | N/A     | [link](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1/RESULTS.md#transformer-only-20-epochs)                                                             |
| Librispeech dev_clean  | N/A     | 2.1     | [link](https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--transformer-lm-4-gpus)             |
| Librispeech dev_other  | N/A     | 5.3     | same as above                                                                                                                                                         |
| Librispeech test_clean | N/A     | 2.5     | same as above                                                                                                                                                         |
| Librispeech test_other | N/A     | 5.5     | same as above                                                                                                                                                         |
| TEDLIUM2 dev           | N/A     | 9.3     | [link](https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md#transformer-large-model--specaug--large-lm)                                          |
| TEDLIUM2 test          | N/A     | 8.1     | same as above                                                                                                                                                         |
| TEDLIUM3 dev           | N/A     | 9.7     | [link](https://github.com/espnet/espnet/blob/master/egs/tedlium3/asr1/RESULTS.md#transformer-elayers12-dlayers6-units2048-8-gpus-specaug--large-lm)                   |
| TEDLIUM3 test          | N/A     | 8.0     | same as above                                                                                                                                                         |
| WSJ dev93              | 3.2     | 7.0     | N/A                                                                                                                                                                   |
| WSJ eval92             | 2.1     | 4.7     | N/A                                                                                                                                                                   |

Note that the performance of the CSJ, HKUST, and Librispeech tasks was significantly improved by using the wide network (#units = 1024) and large subword units if necessary reported by [RWTH](https://arxiv.org/pdf/1805.03294.pdf).

If you want to check the results of the other recipes, please check `egs/<name_of_recipe>/asr1/RESULTS.md`.

</div></details>


### ASR demo

<details><summary>expand</summary><div>

You can recognize speech in a WAV file using pretrained models.
Go to a recipe directory and run `utils/recog_wav.sh` as follows:
```sh
# go to recipe directory and source path of espnet tools
cd egs/tedlium2/asr1 && . ./path.sh
# let's recognize speech!
recog_wav.sh --models tedlium2.transformer.v1 example.wav
```
where `example.wav` is a WAV file to be recognized.
The sampling rate must be consistent with that of data used in training.

Available pretrained models in the demo script are listed as below.

| Model                                                                                            | Notes                                                      |
| :------                                                                                          | :------                                                    |
| [tedlium2.rnn.v1](https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe)            | Streaming decoding based on CTC-based VAD                  |
| [tedlium2.rnn.v2](https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf)            | Streaming decoding based on CTC-based VAD (batch decoding) |
| [tedlium2.transformer.v1](https://drive.google.com/open?id=1heuP2G5YX5u4hERs370eF-1MG2DT50zR)    | Joint-CTC attention Transformer trained on Tedlium 2       |
| [tedlium3.transformer.v1](https://drive.google.com/open?id=1ESVWQp0ZMhenF_Dt1n47suMK8NJ8hm0A)    | Joint-CTC attention Transformer trained on Tedlium 3       |
| [librispeech.transformer.v1](https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6) | Joint-CTC attention Transformer trained on Librispeech     |
| [commonvoice.transformer.v1](https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh) | Joint-CTC attention Transformer trained on CommonVoice     |
| [csj.transformer.v1](https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF)         | Joint-CTC attention Transformer trained on CSJ             |

</div></details>

### ST results

<details><summary>expand</summary><div>

We list 4-gram BLEU of major ST tasks.

#### end-to-end system
| Task | BLEU | Pretrained model |
| ---- | :----: | :----: |
| Fisher-CallHome Spanish fisher_test (Es->En)      | 48.39 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/st1/RESULTS.md#train_spen_lcrm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans) |
| Fisher-CallHome Spanish callhome_evltest (Es->En) | 18.67 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/st1/RESULTS.md#train_spen_lcrm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans) |
| Libri-trans test (En->Fr)                         | 16.70 | [link](https://github.com/espnet/espnet/blob/master/egs/libri_trans/st1/RESULTS.md#train_spfr_lc_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans-1) |
| How2 dev5 (En->Pt)                                | 45.68 | [link](https://github.com/espnet/espnet/blob/master/egs/how2/st1/RESULTS.md#trainpt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans-1) |
| Must-C tst-COMMON (En->De)                        | 22.91 | [link](https://github.com/espnet/espnet/blob/master/egs/must_c/st1/RESULTS.md#train_spen-dede_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans) |
| Mboshi-French dev (Fr->Mboshi)                    | 6.18  | N/A  |

#### cascaded system
| Task | BLEU | Pretrained model |
| ---- | :----: | :----: |
| Fisher-CallHome Spanish fisher_test (Es->En)      | 42.16 | N/A  |
| Fisher-CallHome Spanish callhome_evltest (Es->En) | 19.82 | N/A  |
| Libri-trans test (En->Fr)                         | 16.96 | N/A  |
| How2 dev5 (En->Pt)                                | 44.90 | N/A  |
| Must-C tst-COMMON (En->De)                        | 23.65 | N/A  |

If you want to check the results of the other recipes, please check `egs/<name_of_recipe>/st1/RESULTS.md`.

</div></details>


### ST demo

<details><summary>expand</summary><div>

(**New!**) We made a new real-time E2E-ST + TTS demonstration in Google Colab.
Please access the notebook from the following button and enjoy the real-time speech-to-speech translation!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/st_demo.ipynb)

---

You can translate speech in a WAV file using pretrained models.
Go to a recipe directory and run `utils/translate_wav.sh` as follows:
```sh
# go to recipe directory and source path of espnet tools
cd egs/fisher_callhome_spanish/st1 && . ./path.sh
# download example wav file
wget -O - https://github.com/espnet/espnet/files/4100928/test.wav.tar.gz | tar zxvf -
# let's translate speech!
translate_wav.sh --models fisher_callhome_spanish.transformer.v1.es-en test.wav
```
where `test.wav` is a WAV file to be translated.
The sampling rate must be consistent with that of data used in training.

Available pretrained models in the demo script are listed as below.

| Model                                                                                            | Notes                                                      |
| :------                                                                                          | :------                                                    |
| [fisher_callhome_spanish.transformer.v1](https://drive.google.com/open?id=1hawp5ZLw4_SIHIT3edglxbKIIkPVe8n3)            | Transformer-ST trained on Fisher-CallHome Spanish Es->En                  |

</div></details>


### MT results

<details><summary>expand</summary><div>

| Task | BLEU | Pretrained model |
| ---- | :----: | :----: |
| Fisher-CallHome Spanish fisher_test (Es->En)      | 61.45 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/mt1/RESULTS.md#trainen_lcrm_lcrm_pytorch_train_pytorch_transformer_bpe_bpe1000) |
| Fisher-CallHome Spanish callhome_evltest (Es->En) | 29.86 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/mt1/RESULTS.md#trainen_lcrm_lcrm_pytorch_train_pytorch_transformer_bpe_bpe1000) |
| Libri-trans test (En->Fr)                         | 18.09 | [link](https://github.com/espnet/espnet/blob/master/egs/libri_trans/mt1/RESULTS.md#trainfr_lcrm_tc_pytorch_train_pytorch_transformer_bpe1000) |
| How2 dev5 (En->Pt)                                | 58.61 | [link](https://github.com/espnet/espnet/blob/master/egs/how2/mt1/RESULTS.md#trainpt_tc_tc_pytorch_train_pytorch_transformer_bpe8000) |
| Must-C tst-COMMON (En->De)                        | 27.63 | [link](https://github.com/espnet/espnet/blob/master/egs/must_c/mt1/RESULTS.md#summary-4-gram-bleu) |
| IWSLT'14 test2014 (En->De)                        | 24.70 | [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result) |
| IWSLT'14 test2014 (De->En)                        | 29.22 | [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result) |
| IWSLT'16 test2014 (En->De)                        | 24.05 | [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result) |
| IWSLT'16 test2014 (De->En)                        | 29.13 | [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result) |

</div></details>

### TTS results

<details><summary>expand</summary><div>

You can listen to our samples in demo HP [espnet-tts-sample](https://espnet.github.io/espnet-tts-sample/).
Here we list some notable ones:

- [Single English speaker Tacotron2](https://drive.google.com/open?id=18JgsOCWiP_JkhONasTplnHS7yaF_konr)
- [Single Japanese speaker Tacotron2](https://drive.google.com/open?id=1fEgS4-K4dtgVxwI4Pr7uOA1h4PE-zN7f)
- [Single other language speaker Tacotron2](https://drive.google.com/open?id=1q_66kyxVZGU99g8Xb5a0Q8yZ1YVm2tN0)
- [Multi English speaker Tacotron2](https://drive.google.com/open?id=18S_B8Ogogij34rIfJOeNF8D--uG7amz2)
- [Single English speaker Transformer](https://drive.google.com/open?id=14EboYVsMVcAq__dFP1p6lyoZtdobIL1X)
- [Single English speaker FastSpeech](https://drive.google.com/open?id=1PSxs1VauIndwi8d5hJmZlppGRVu2zuy5)
- [Multi English speaker Transformer](https://drive.google.com/open?id=1_vrdqjM43DdN1Qz7HJkvMQ6lCMmWLeGp)
- [Single Italian speaker FastSpeech](https://drive.google.com/open?id=13I5V2w7deYFX4DlVk1-0JfaXmUR2rNOv)
- [Single Mandarin speaker Transformer](https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD)
- [Single Mandarin speaker FastSpeech](https://drive.google.com/open?id=1Ol_048Tuy6BgvYm1RpjhOX4HfhUeBqdK)
- [Multi Japanese speaker Transformer](https://drive.google.com/open?id=1fFMQDF6NV5Ysz48QLFYE8fEvbAxCsMBw)
- [Single English speaker models with Parallel WaveGAN](https://drive.google.com/open?id=1HvB0_LDf1PVinJdehiuCt5gWmXGguqtx)
- [Single English speaker knowledge distillation-based FastSpeech (New!)](https://drive.google.com/open?id=1wG-Y0itVYalxuLAHdkAHO7w1CWFfRPF4)

You can download all of the pretrained models and generated samples:
- [All of the pretrained E2E-TTS models](https://drive.google.com/open?id=1k9RRyc06Zl0mM2A7mi-hxNiNMFb_YzTF)
- [All of the generated samples](https://drive.google.com/open?id=1bQGuqH92xuxOX__reWLP4-cif0cbpMLX)

Note that in the generated samples we use the following vocoders: Griffin-Lim (**GL**), WaveNet vocoder (**WaveNet**), Parallel WaveGAN (**ParallelWaveGAN**), and MelGAN (**MelGAN**).
The neural vocoders are based on following repositories.
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN): Parallel WaveGAN / MelGAN / Multi-band MelGAN
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder): 16 bit mixture of Logistics WaveNet vocoder
- [kan-bayashi/PytorchWaveNetVocoder](https://github.com/kan-bayashi/PytorchWaveNetVocoder): 8 bit Softmax WaveNet Vocoder with the noise shaping

If you want to build your own neural vocoder, please check the above repositories.
[kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) provides [the manual](https://github.com/kan-bayashi/ParallelWaveGAN#decoding-with-espnet-tts-models-features) about how to decode ESPnet-TTS model's features with neural vocoders. Please check it.

Here we list all of the pretrained neural vocoders. Please download and enjoy the generation of high quality speech!

| Model link                                                                                              | Lang  | Fs [Hz] | Mel range [Hz] | FFT / Shift / Win [pt] | Model type                                                              |
| :------                                                                                                 | :---: | :----:  | :--------:     | :---------------:      | :------                                                                 |
| [ljspeech.wavenet.softmax.ns.v1](https://drive.google.com/open?id=1eA1VcRS9jzFa-DovyTgJLQ_jmwOLIi8L)    | EN    | 22.05k  | None           | 1024 / 256 / None      | [Softmax WaveNet](https://github.com/kan-bayashi/PytorchWaveNetVocoder) |
| [ljspeech.wavenet.mol.v1](https://drive.google.com/open?id=1sY7gEUg39QaO1szuN62-Llst9TrFno2t)           | EN    | 22.05k  | None           | 1024 / 256 / None      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [ljspeech.parallel_wavegan.v1](https://drive.google.com/open?id=1tv9GKyRT4CDsvUWKwH3s_OfXkiTi0gw7)      | EN    | 22.05k  | None           | 1024 / 256 / None      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [ljspeech.wavenet.mol.v2](https://drive.google.com/open?id=1es2HuKUeKVtEdq6YDtAsLNpqCy4fhIXr)           | EN    | 22.05k  | 80-7600        | 1024 / 256 / None      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [ljspeech.parallel_wavegan.v2](https://drive.google.com/open?id=1Grn7X9wD35UcDJ5F7chwdTqTa4U7DeVB)      | EN    | 22.05k  | 80-7600        | 1024 / 256 / None      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [ljspeech.melgan.v1 (EXPERIMENTAL)](https://drive.google.com/open?id=1ipPWYl8FBNRlBFaKj1-i23eQpW_W_YcR) | EN    | 22.05k  | 80-7600        | 1024 / 256 / None      | [MelGAN](https://github.com/kan-bayashi/ParallelWaveGAN)                |
| [ljspeech.melgan.v3 (EXPERIMENTAL)](https://drive.google.com/open?id=1_a8faVA5OGCzIcJNw4blQYjfG4oA9VEt) | EN    | 22.05k  | 80-7600        | 1024 / 256 / None      | [MelGAN](https://github.com/kan-bayashi/ParallelWaveGAN)                |
| [libritts.wavenet.mol.v1](https://drive.google.com/open?id=1jHUUmQFjWiQGyDd7ZeiCThSjjpbF_B4h)           | EN    | 24k     | None           | 1024 / 256 / None      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [jsut.wavenet.mol.v1](https://drive.google.com/open?id=187xvyNbmJVZ0EZ1XHCdyjZHTXK9EcfkK)               | JP    | 24k     | 80-7600        | 2048 / 300 / 1200      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [jsut.parallel_wavegan.v1](https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM)          | JP    | 24k     | 80-7600        | 2048 / 300 / 1200      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [csmsc.wavenet.mol.v1](https://drive.google.com/open?id=1PsjFRV5eUP0HHwBaRYya9smKy5ghXKzj)              | ZH    | 24k     | 80-7600        | 2048 / 300 / 1200      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [csmsc.parallel_wavegan.v1](https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy)         | ZH    | 24k     | 80-7600        | 2048 / 300 / 1200      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |

If you want to use the above pretrained vocoders, please exactly match the feature setting with them.

</div></details>

### TTS demo

<details><summary>expand</summary><div>

(**New!**) We made a new real-time E2E-TTS demonstration in Google Colab.
Please access the notebook from the following button and enjoy the real-time synthesis!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb)

---

You can synthesize speech in a TXT file using pretrained models.
Go to a recipe directory and run `utils/synth_wav.sh` as follows:

```sh
# go to recipe directory and source path of espnet tools
cd egs/ljspeech/tts1 && . ./path.sh
# we use upper-case char sequence for the default model.
echo "THIS IS A DEMONSTRATION OF TEXT TO SPEECH." > example.txt
# let's synthesize speech!
synth_wav.sh example.txt

# also you can use multiple sentences
echo "THIS IS A DEMONSTRATION OF TEXT TO SPEECH." > example_multi.txt
echo "TEXT TO SPEECH IS A TECHQNIQUE TO CONVERT TEXT INTO SPEECH." >> example_multi.txt
synth_wav.sh example_multi.txt
```

You can change the pretrained model as follows:

```sh
synth_wav.sh --models ljspeech.fastspeech.v1 example.txt
```

Waveform synthesis is performed with Griffin-Lim algorithm and neural vocoders (WaveNet and ParallelWaveGAN).
You can change the pretrained vocoder model as follows:

```sh
synth_wav.sh --vocoder_models ljspeech.wavenet.mol.v1 example.txt
```

WaveNet vocoder provides very high quality speech but it takes time to generate.

> **Important Note**:
>
> This code does not include text frontend part.
> Please clean the input text manually.
> Also, you need to modify feature configuration according to the model.
> Default setting is for ljspeech models, so if you want to use other pretrained models, please modify the parameters by yourself.
> For our provided models, you can find them in the below table.
>
> If you are beginner, instead of this script, I strongly recommend trying the [colab notebook](https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb) at first, which includes all of the procedure from text frontend, feature generation, and waveform generation.

Available pretrained models in the demo script are listed as follows:

| Model link                                                                                    | Lang  | Fs [Hz] | Mel range [Hz] | FFT / Shift / Win [pt] | Input  | R   | Model type                                  |
| :------                                                                                       | :---: | :----:  | :--------:     | :---------------:      | :---:  | :-: | :------                                     |
| [ljspeech.tacotron2.v1](https://drive.google.com/open?id=1dKzdaDpOkpx7kWZnvrvx2De7eZEdPHZs)   | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 2   | Tacotron 2                                  |
| [ljspeech.tacotron2.v2](https://drive.google.com/open?id=11T9qw8rJlYzUdXvFjkjQjYrp3iGfQ15h)   | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 1   | Tacotron 2 + forward attention              |
| [ljspeech.tacotron2.v3](https://drive.google.com/open?id=1hiZn14ITUDM1nkn-GkaN_M3oaTOUcn1n)   | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 1   | Tacotron 2 + guided attention loss          |
| [ljspeech.transformer.v1](https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS) | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 1   | Deep Transformer                            |
| [ljspeech.transformer.v2](https://drive.google.com/open?id=1xxAwPuUph23RnlC5gym7qDM02ZCW9Unp) | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 3   | Shallow Transformer                         |
| [ljspeech.transformer.v3](https://drive.google.com/open?id=1M_w7nxI6AfbtSHpMO-exILnAc_aUYvXP) | EN    | 22.05k  | None           | 1024 / 256 / None      | phn    | 1   | Deep Transformer                            |
| [ljspeech.fastspeech.v1](https://drive.google.com/open?id=17RUNFLP4SSTbGA01xWRJo7RkR876xM0i)  | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 1   | FF-Transformer                              |
| [ljspeech.fastspeech.v2](https://drive.google.com/open?id=1zD-2GMrWM3thaDpS3h3rkTU4jIC0wc5B)  | EN    | 22.05k  | None           | 1024 / 256 / None      | char   | 1   | FF-Transformer + CNN in FFT block           |
| [ljspeech.fastspeech.v3](https://drive.google.com/open?id=1W86YEQ6KbuUTIvVURLqKtSNqe_eI2GDN)  | EN    | 22.05k  | None           | 1024 / 256 / None      | phn    | 1   | FF-Transformer + CNN in FFT block + postnet |
| [libritts.tacotron2.v1](https://drive.google.com/open?id=1iAXwC0AuWusa9AcFeUVkcNLG0I-hnSr3)   | EN    | 24k     | 80-7600        | 1024 / 256 / None      | char   | 2   | Multi-speaker Tacotron 2                    |
| [libritts.transformer.v1](https://drive.google.com/open?id=1Xj73mDPuuPH8GsyNO8GnOC3mn0_OK4g3) | EN    | 24k     | 80-7600        | 1024 / 256 / None      | char   | 2   | Multi-speaker Transformer                   |
| [jsut.tacotron2](https://drive.google.com/open?id=1kp5M4VvmagDmYckFJa78WGqh1drb_P9t)          | JP    | 24k     | 80-7600        | 2048 / 300 / 1200      | phn    | 2   | Tacotron 2                                  |
| [jsut.transformer](https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD)        | JP    | 24k     | 80-7600        | 2048 / 300 / 1200      | phn    | 3   | Shallow Transformer                         |
| [csmsc.transformer.v1](https://drive.google.com/open?id=1bTSygvonv5TS6-iuYsOIUWpN2atGnyhZ)    | ZH    | 24k     | 80-7600        | 2048 / 300 / 1200      | pinyin | 1   | Deep Transformer                            |
| [csmsc.fastspeech.v3](https://drive.google.com/open?id=1T8thxkAxjGFPXPWPTcKLvHnd6lG0-82R)     | ZH    | 24k     | 80-7600        | 2048 / 300 / 1200      | pinyin | 1   | FF-Transformer + CNN in FFT block + postnet |

Available pretrained vocoder models in the demo script are listed as follows:

| Model link                                                                                           | Lang  | Fs [Hz] | Mel range [Hz] | FFT / Shift / Win [pt] | Model type                                                              |
| :------                                                                                              | :---: | :----:  | :--------:     | :---------------:      | :------                                                                 |
| [ljspeech.wavenet.softmax.ns.v1](https://drive.google.com/open?id=1eA1VcRS9jzFa-DovyTgJLQ_jmwOLIi8L) | EN    | 22.05k  | None           | 1024 / 256 / None      | [Softmax WaveNet](https://github.com/kan-bayashi/PytorchWaveNetVocoder) |
| [ljspeech.wavenet.mol.v1](https://drive.google.com/open?id=1sY7gEUg39QaO1szuN62-Llst9TrFno2t)        | EN    | 22.05k  | None           | 1024 / 256 / None      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [ljspeech.parallel_wavegan.v1](https://drive.google.com/open?id=1tv9GKyRT4CDsvUWKwH3s_OfXkiTi0gw7)   | EN    | 22.05k  | None           | 1024 / 256 / None      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [libritts.wavenet.mol.v1](https://drive.google.com/open?id=1jHUUmQFjWiQGyDd7ZeiCThSjjpbF_B4h)        | EN    | 24k     | None           | 1024 / 256 / None      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [jsut.wavenet.mol.v1](https://drive.google.com/open?id=187xvyNbmJVZ0EZ1XHCdyjZHTXK9EcfkK)            | JP    | 24k     | 80-7600        | 2048 / 300 / 1200      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [jsut.parallel_wavegan.v1](https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM)       | JP    | 24k     | 80-7600        | 2048 / 300 / 1200      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [csmsc.wavenet.mol.v1](https://drive.google.com/open?id=1PsjFRV5eUP0HHwBaRYya9smKy5ghXKzj)           | ZH    | 24k     | 80-7600        | 2048 / 300 / 1200      | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [csmsc.parallel_wavegan.v1](https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy)      | ZH    | 24k     | 80-7600        | 2048 / 300 / 1200      | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |

</div></details>

### VC results

The [Voice Conversion Challenge 2020](http://www.vc-challenge.org/) (VCC2020) adopts ESPnet to build an end-to-end based baseline system. In VCC2020, the objective is intra/cross lingual nonparallel VC. A cascade method of ASR+TTS is developed.  
You can download converted samples [here](https://drive.google.com/drive/folders/1oeZo83GrOgtqxGwF7KagzIrfjr8X59Ue?usp=sharing).

## References

[1] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-End Speech Processing Toolkit," *Proc. Interspeech'18*, pp. 2207-2211 (2018)

[2] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[3] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

## Citations

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={ESPnet: End-to-End Speech Processing Toolkit},
  year=2018,
  booktitle={Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@misc{hayashi2019espnettts,
    title={ESPnet-TTS: Unified, Reproducible, and Integratable Open Source End-to-End Text-to-Speech Toolkit},
    author={Tomoki Hayashi and Ryuichi Yamamoto and Katsuki Inoue and Takenori Yoshimura and Shinji Watanabe and Tomoki Toda and Kazuya Takeda and Yu Zhang and Xu Tan},
    year={2019},
    eprint={1910.10909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
@article{inaguma2020espnet,
  title={ESPnet-ST: All-in-One Speech Translation Toolkit},
  author={Inaguma, Hirofumi and Kiyono, Shun and Duh, Kevin and Karita, Shigeki and Soplin, Nelson Enrique Yalta and Hayashi, Tomoki and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2004.10234},
  year={2020}
}
```
