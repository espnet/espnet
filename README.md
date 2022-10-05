<div align="left"><img src="doc/image/espnet_logo1.png" width="550"/></div>

# ESPnet: end-to-end speech processing toolkit

|system/pytorch ver.|1.4.0|1.5.1|1.6.0|1.7.1|1.8.1|1.9.1|1.10.2|1.11.0|1.12.1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ubuntu20/python3.9/pip|||||||||[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|
|ubuntu20/python3.8/pip|||||||||[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|
|ubuntu18/python3.7/pip|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|[![Github Actions](https://github.com/espnet/espnet/workflows/CI/badge.svg)](https://github.com/espnet/espnet/actions)|
|debian9/python3.7/conda|||||||||[![debian9](https://github.com/espnet/espnet/workflows/debian9/badge.svg)](https://github.com/espnet/espnet/actions?query=workflow%3Adebian9)|
|centos7/python3.7/conda|||||||||[![centos7](https://github.com/espnet/espnet/workflows/centos7/badge.svg)](https://github.com/espnet/espnet/actions?query=workflow%3Acentos7)|
|doc/python3.8|||||||||[![doc](https://github.com/espnet/espnet/workflows/doc/badge.svg)](https://github.com/espnet/espnet/actions?query=workflow%3Adoc)|


[![PyPI version](https://badge.fury.io/py/espnet.svg)](https://badge.fury.io/py/espnet)
[![Python Versions](https://img.shields.io/pypi/pyversions/espnet.svg)](https://pypi.org/project/espnet/)
[![Downloads](https://pepy.tech/badge/espnet)](https://pepy.tech/project/espnet)
[![GitHub license](https://img.shields.io/github/license/espnet/espnet.svg)](https://github.com/espnet/espnet)
[![codecov](https://codecov.io/gh/espnet/espnet/branch/master/graph/badge.svg)](https://codecov.io/gh/espnet/espnet)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Mergify Status](https://img.shields.io/endpoint.svg?url=https://api.mergify.com/v1/badges/espnet/espnet&style=flat)](https://mergify.com)
[![Gitter](https://badges.gitter.im/espnet-en/community.svg)](https://gitter.im/espnet-en/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[**Docs**](https://espnet.github.io/espnet/)
| [**Example**](https://github.com/espnet/espnet/tree/master/egs)
| [**Example (ESPnet2)**](https://github.com/espnet/espnet/tree/master/egs2)
| [**Docker**](https://github.com/espnet/espnet/tree/master/docker)
| [**Notebook**](https://github.com/espnet/notebook)

ESPnet is an end-to-end speech processing toolkit covering end-to-end speech recognition, text-to-speech, speech translation, speech enhancement, speaker diarization, spoken language understanding, and so on.
ESPnet uses [pytorch](http://pytorch.org/) as a deep learning engine and also follows [Kaldi](http://kaldi-asr.org/) style data processing, feature extraction/format, and recipes to provide a complete setup for various speech processing experiments.

## Tutorial Series
- 2019 Tutorial at Interspeech
  - [Material](https://github.com/espnet/interspeech2019-tutorial)
- 2021 Tutorial at CMU
  - [Online video](https://youtu.be/2mRz3wH1vd0)
  - [Material](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_tutorial_2021_CMU_11751_18781.ipynb)
- 2022 Tutorial at CMU
  - Usage of ESPnet (ASR as an example)
    - [Online video]()
    - [Material]()
  - Add new models/tasks to ESPnet
    - [Online video]()
    - [Material](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_new_task_tutorial_CMU_11751_18781_Fall2022.ipynb)


## Key Features

### Kaldi style complete recipe
- Support numbers of `ASR` recipes (WSJ, Switchboard, CHiME-4/5, Librispeech, TED, CSJ, AMI, HKUST, Voxforge, REVERB, etc.)
- Support numbers of `TTS` recipes with a similar manner to the ASR recipe (LJSpeech, LibriTTS, M-AILABS, etc.)
- Support numbers of `ST` recipes (Fisher-CallHome Spanish, Libri-trans, IWSLT'18, How2, Must-C, Mboshi-French, etc.)
- Support numbers of `MT` recipes (IWSLT'14, IWSLT'16, the above ST recipes etc.)
- Support numbers of `SLU` recipes (CATSLU-MAPS, FSC, Grabo, IEMOCAP, JDCINAL, SNIPS, SLURP, SWBD-DA, etc.)
- Support numbers of `SE/SS` recipes (DNS-IS2020, LibriMix, SMS-WSJ, VCTK-noisyreverb, WHAM!, WHAMR!, WSJ-2mix, etc.)
- Support voice conversion recipe (VCC2020 baseline)
- Support speaker diarization recipe (mini_librispeech, librimix)
- Support singing voice synthesis recipe (ofuton_p_utagoe_db)

### ASR: Automatic Speech Recognition
- **State-of-the-art performance** in several ASR benchmarks (comparable/superior to hybrid DNN/HMM and CTC)
- **Hybrid CTC/attention** based end-to-end ASR
  - Fast/accurate training with CTC/attention multitask training
  - CTC/attention joint decoding to boost monotonic alignment decoding
  - Encoder: VGG-like CNN + BiRNN (LSTM/GRU), sub-sampling BiRNN (LSTM/GRU), Transformer, Conformer or [Branchformer](https://proceedings.mlr.press/v162/peng22a.html)
- Attention: Dot product, location-aware attention, variants of multi-head
- Incorporate RNNLM/LSTMLM/TransformerLM/N-gram trained only with text data
- Batch GPU decoding
- Data augmentation
- **Transducer** based end-to-end ASR
  - Architecture:
    - RNN-based encoder and decoder.
    - Custom encoder and decoder supporting Transformer, Conformer (encoder), 1D Conv / TDNN (encoder) and causal 1D Conv (decoder) blocks.
    - VGG2L (RNN/custom encoder) and Conv2D (custom encoder) bottlenecks.
  - Search algorithms:
    - Greedy search constrained to one emission by timestep.
    - Default beam search algorithm [[Graves, 2012]](https://arxiv.org/abs/1211.3711) without prefix search.
    - Alignment-Length Synchronous decoding [[Saon et al., 2020]](https://ieeexplore.ieee.org/abstract/document/9053040).
    - Time Synchronous Decoding [[Saon et al., 2020]](https://ieeexplore.ieee.org/abstract/document/9053040).
    - N-step Constrained beam search modified from [[Kim et al., 2020]](https://arxiv.org/abs/2002.03577).
    - modified Adaptive Expansion Search based on [[Kim et al., 2021]](https://ieeexplore.ieee.org/abstract/document/9250505) and NSC.
  - Features:
    - Multi-task learning with various auxiliary losses:
      - Encoder: CTC, auxiliary Transducer and symmetric KL divergence.
      - Decoder: cross-entropy w/ label smoothing.
    - Transfer learning with acoustic model and/or language model.
    - Training with FastEmit regularization method [[Yu et al., 2021]](https://arxiv.org/abs/2010.11148).
  > Please refer to the [tutorial page](https://espnet.github.io/espnet/tutorial.html#transducer) for complete documentation.
- CTC segmentation
- Non-autoregressive model based on Mask-CTC
- ASR examples for supporting endangered language documentation (Please refer to egs/puebla_nahuatl and egs/yoloxochitl_mixtec for details)
- Wav2Vec2.0 pretrained model as Encoder, imported from [FairSeq](https://github.com/pytorch/fairseq/tree/master/fairseq).
- Self-supervised learning representations as features, using upstream models in [S3PRL](https://github.com/s3prl/s3prl) in frontend.
  - Set `frontend` to be `s3prl`
  - Select any upstream model by setting the `frontend_conf` to the corresponding name.
- Transfer Learning :
  - easy usage and transfers from models previously trained by your group, or models from [ESPnet huggingface repository](https://huggingface.co/espnet).
  - [Documentation](https://github.com/espnet/espnet/tree/master/egs2/mini_an4/asr1/transfer_learning.md) and [toy example runnable on colab](https://github.com/espnet/notebook/blob/master/espnet2_asr_transfer_learning_demo.ipynb).
- Streaming Transformer/Conformer ASR with blockwise synchronous beam search.
- Restricted Self-Attention based on [Longformer](https://arxiv.org/abs/2004.05150) as an encoder for long sequences

Demonstration
- Real-time ASR demo with ESPnet2  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_asr_realtime_demo.ipynb)
- [Gradio](https://github.com/gradio-app/gradio) Web Demo on [Huggingface Spaces](https://huggingface.co/docs/hub/spaces). Check out the [Web Demo](https://huggingface.co/spaces/akhaliq/espnet2_asr)
- Streaming Transformer ASR [Local Demo](https://github.com/espnet/notebook/blob/master/espnet2_streaming_asr_demo.ipynb) with ESPnet2.

### TTS: Text-to-speech
- Architecture
    - Tacotron2
    - Transformer-TTS
    - FastSpeech
    - FastSpeech2
    - Conformer FastSpeech & FastSpeech2
    - VITS
    - JETS
- Multi-speaker & multi-language extention
    - Pretrained speaker embedding (e.g., X-vector)
    - Speaker ID embedding
    - Language ID embedding
    - Global style token (GST) embedding
    - Mix of the above embeddings
- End-to-end training
    - End-to-end text-to-wav model (e.g., VITS, JETS, etc.)
    - Joint training of text2mel and vocoder
- Various language support
    - En / Jp / Zn / De / Ru / And more...
- Integration with neural vocoders
    - Parallel WaveGAN
    - MelGAN
    - Multi-band MelGAN
    - HiFiGAN
    - StyleMelGAN
    - Mix of the above models

Demonstration
- Real-time TTS demo with ESPnet2  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_tts_realtime_demo.ipynb)
- Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/ESPnet2-TTS)

To train the neural vocoder, please check the following repositories:
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)

> **NOTE**:
> - We are moving on ESPnet2-based development for TTS.
> - The use of ESPnet1-TTS is deprecated, please use [ESPnet2-TTS](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/tts1).

### SE: Speech enhancement (and separation)

- Single-speaker speech enhancement
- Multi-speaker speech separation
- Unified encoder-separator-decoder structure for time-domain and frequency-domain models
  - Encoder/Decoder: STFT/iSTFT, Convolution/Transposed-Convolution
  - Separators: BLSTM, Transformer, Conformer, [TasNet](https://arxiv.org/abs/1809.07454), [DPRNN](https://arxiv.org/abs/1910.06379), [SkiM](https://arxiv.org/abs/2201.10800), [SVoice](https://arxiv.org/abs/2011.02329), [DC-CRN](https://web.cse.ohio-state.edu/~wang.77/papers/TZW.taslp21.pdf), [DCCRN](https://arxiv.org/abs/2008.00264), [Deep Clustering](https://ieeexplore.ieee.org/document/7471631), [Deep Attractor Network](https://pubmed.ncbi.nlm.nih.gov/29430212/), [FaSNet](https://arxiv.org/abs/1909.13387), [iFaSNet](https://arxiv.org/abs/1910.14104), Neural Beamformers, etc.
- Flexible ASR integration: working as an individual task or as the ASR frontend
- Easy to import pretrained models from [Asteroid](https://github.com/asteroid-team/asteroid)
  - Both the pre-trained models from Asteroid and the specific configuration are supported.

Demonstration
- Interactive SE demo with ESPnet2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fjRJCh96SoYLZPRxsjF9VDv4Q2VoIckI?usp=sharing)

### ST: Speech Translation & MT: Machine Translation
- **State-of-the-art performance** in several ST benchmarks (comparable/superior to cascaded ASR and MT)
- Transformer based end-to-end ST (new!)
- Transformer based end-to-end MT (new!)

### VC: Voice conversion
- Transformer and Tacotron2 based parallel VC using melspectrogram (new!)
- End-to-end VC based on cascaded ASR+TTS (Baseline system for Voice Conversion Challenge 2020!)

### SLU: Spoken Language Understanding
- Architecture
    - Transformer based Encoder
    - Conformer based Encoder
    - [Branchformer](https://proceedings.mlr.press/v162/peng22a.html) based Encoder
    - RNN based Decoder
    - Transformer based Decoder
- Support Multitasking with ASR
    - Predict both intent and ASR transcript
- Support Multitasking with NLU
    - Deliberation encoder based 2 pass model
- Support using pretrained ASR models
    - Hubert
    - Wav2vec2
    - VQ-APC
    - TERA and more ...
- Support using pretrained NLP models
    - BERT
    - MPNet And more...
- Various language support
    - En / Jp / Zn / Nl / And more...
- Supports using context from previous utterances
- Supports using other tasks like SE in pipeline manner
- Supports Two Pass SLU that combines audio and ASR transcript
Demonstration
- Performing noisy spoken language understanding using speech enhancement model followed by spoken language understanding model.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14nCrJ05vJcQX0cJuXjbMVFWUHJ3Wfb6N?usp=sharing)
- Performing two pass spoken language understanding where the second pass model attends on both acoustic and semantic information.  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p2cbGIPpIIcynuDl4ZVHDpmNPl8Nh_ci?usp=sharing)
- Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See SLU demo on multiple languages: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Siddhant/ESPnet2-SLU)


### SUM: Speech Summarization
- End to End Speech Summarization Recipe for Instructional Videos using Restricted Self-Attention [[Sharma et al., 2022]](https://arxiv.org/abs/2110.06263)

### SVS: Singing Voice Synthesis
- Framework merge from [Muskits](https://github.com/SJTMusicTeam/Muskits)
- Architecture
  - RNN-based non-autoregressive model
  - Xiaoice
  - Sequence-to-sequence Transformer (with GLU-based encoder)
  - MLP singer
  - Tacotron-singing (in progress)
  - DiffSinger (to be published)
  - VISinger (in progress)
- Support multi-speaker & multilingual singing synthesis
  - Speaker ID embedding
  - Language ID embedding
  - Global sytle token (GST) embedding
- Various language support
  - Jp / En / Kr / Zh
- Tight integration with neural vocoders (the same as TTS)

### DNN Framework
- Flexible network architecture thanks to chainer and pytorch
- Flexible front-end processing thanks to [kaldiio](https://github.com/nttcslab-sp/kaldiio) and HDF5 support
- Tensorboard based monitoring

### ESPnet2
See [ESPnet2](https://espnet.github.io/espnet/espnet2_tutorial.html).

- Independent from Kaldi/Chainer, unlike ESPnet1
- On the fly feature extraction and text processing when training
- Supporting DistributedDataParallel and DaraParallel both
- Supporting multiple nodes training and integrated with [Slurm](https://slurm.schedmd.com/) or MPI
- Supporting Sharded Training provided by [fairscale](https://github.com/facebookresearch/fairscale)
- A template recipe which can be applied for all corpora
- Possible to train any size of corpus without CPU memory error
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)
- Integrated with [wandb](https://espnet.github.io/espnet/espnet2_training_option.html#weights-biases-integration)

## Installation
- If you intend to do full experiments including DNN training, then see [Installation](https://espnet.github.io/espnet/installation.html).
- If you just need the Python module only:
    ```sh
    # We recommend you installing pytorch before installing espnet following https://pytorch.org/get-started/locally/
    pip install espnet
    # To install latest
    # pip install git+https://github.com/espnet/espnet
    # To install additional packages
    # pip install "espnet[all]"
    ```

    If you'll use ESPnet1, please install chainer and cupy.

    ```sh
    pip install chainer==6.0.0 cupy==6.0.0    # [Option]
    ```

    You might need to install some packages depending on each task. We prepared various installation scripts at [tools/installers](tools/installers).

- (ESPnet2) Once installed, run `wandb login` and set `--use_wandb true` to enable tracking runs using W&B.

## Usage
See [Usage](https://espnet.github.io/espnet/tutorial.html).

## Docker Container

go to [docker/](docker/) and follow [instructions](https://espnet.github.io/espnet/docker.html).

## Contribution
Thank you for taking times for ESPnet! Any contributions to ESPnet are welcome and feel free to ask any questions or requests to [issues](https://github.com/espnet/espnet/issues).
If it's the first contribution to ESPnet for you,  please follow the [contribution guide](CONTRIBUTING.md).

## Results and demo

You can find useful tutorials and demos in [Interspeech 2019 Tutorial](https://github.com/espnet/interspeech2019-tutorial)

### ASR results

<details><summary>expand</summary><div>


We list the character error rate (CER) and word error rate (WER) of major ASR tasks.

| Task                                                              |     CER (%)     |     WER (%)     |                                                                              Pretrained model                                                                               |
| ----------------------------------------------------------------- | :-------------: | :-------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Aishell dev/test                                                  |     4.6/5.1     |       N/A       |                [link](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#conformer-kernel-size--15--specaugment--lm-weight--00-result)                |
| **ESPnet2** Aishell dev/test                                      |     4.4/4.7     |       N/A       |                [link](https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1#conformer--specaug--speed-perturbation-featsraw-n_fft512-hop_length128)                |
| Common Voice dev/test                                             |     1.7/1.8     |     2.2/2.3     |    [link](https://github.com/espnet/espnet/blob/master/egs/commonvoice/asr1/RESULTS.md#first-results-default-pytorch-transformer-setting-with-bpe-100-epochs-single-gpu)    |
| CSJ eval1/eval2/eval3                                             |   5.7/3.8/4.2   |       N/A       |                 [link](https://github.com/espnet/espnet/blob/master/egs/csj/asr1/RESULTS.md#pytorch-backend-transformer-without-any-hyperparameter-tuning)                  |
| **ESPnet2** CSJ eval1/eval2/eval3                                 |   4.5/3.3/3.6   |       N/A       |                                        [link](https://github.com/espnet/espnet/tree/master/egs2/csj/asr1#initial-conformer-results)                                         |
| HKUST dev                                                         |      23.5       |       N/A       |                                  [link](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1/RESULTS.md#transformer-only-20-epochs)                                  |
| **ESPnet2** HKUST dev                                             |      21.2       |       N/A       |                                    [link](https://github.com/espnet/espnet/tree/master/egs2/hkust/asr1#transformer-asr--transformer-lm)                                     |
| Librispeech dev_clean/dev_other/test_clean/test_other             |       N/A       | 1.9/4.9/2.1/4.9 | [link](https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-conformer-with-specaug--speed-perturbation-8-gpus--transformer-lm-4-gpus) |
| **ESPnet2** Librispeech dev_clean/dev_other/test_clean/test_other | 0.6/1.5/0.6/1.4 | 1.7/3.4/1.8/3.6 |    [link](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#self-supervised-learning-features-hubert_large_ll60k-conformer-utt_mvn-with-transformer-lm)    |
| Switchboard (eval2000) callhm/swbd                                |       N/A       |    14.0/6.8     |          [link](https://github.com/espnet/espnet/blob/master/egs/swbd/asr1/RESULTS.md#conformer-with-bpe-2000-specaug-speed-perturbation-transformer-lm-decoding)           |
| TEDLIUM2 dev/test                                                 |       N/A       |     8.6/7.2     |                 [link](https://github.com/espnet/espnet/blob/master/egs/tedlium2/asr1/RESULTS.md#conformer-large-model--specaug--speed-perturbation--rnnlm)                 |
| TEDLIUM3 dev/test                                                 |       N/A       |     9.6/7.6     |                                              [link](https://github.com/espnet/espnet/blob/master/egs/tedlium3/asr1/RESULTS.md)                                              |
| WSJ dev93/eval92                                                  |     3.2/2.1     |     7.0/4.7     |                                                                                     N/A                                                                                     |
| **ESPnet2** WSJ dev93/eval92                                      |     1.1/0.8     |     2.8/1.8     |       [link](https://github.com/espnet/espnet/tree/master/egs2/wsj/asr1#self-supervised-learning-features-wav2vec2_large_ll60k-conformer-utt_mvn-with-transformer-lm)       |

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
| :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| [tedlium2.rnn.v1](https://drive.google.com/open?id=1UqIY6WJMZ4sxNxSugUqp3mrGb3j6h7xe)            | Streaming decoding based on CTC-based VAD                  |
| [tedlium2.rnn.v2](https://drive.google.com/open?id=1cac5Uc09lJrCYfWkLQsF8eapQcxZnYdf)            | Streaming decoding based on CTC-based VAD (batch decoding) |
| [tedlium2.transformer.v1](https://drive.google.com/open?id=1cVeSOYY1twOfL9Gns7Z3ZDnkrJqNwPow)    | Joint-CTC attention Transformer trained on Tedlium 2       |
| [tedlium3.transformer.v1](https://drive.google.com/open?id=1zcPglHAKILwVgfACoMWWERiyIquzSYuU)    | Joint-CTC attention Transformer trained on Tedlium 3       |
| [librispeech.transformer.v1](https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6) | Joint-CTC attention Transformer trained on Librispeech     |
| [commonvoice.transformer.v1](https://drive.google.com/open?id=1tWccl6aYU67kbtkm8jv5H6xayqg1rzjh) | Joint-CTC attention Transformer trained on CommonVoice     |
| [csj.transformer.v1](https://drive.google.com/open?id=120nUQcSsKeY5dpyMWw_kI33ooMRGT2uF)         | Joint-CTC attention Transformer trained on CSJ             |
| [csj.rnn.v1](https://drive.google.com/open?id=1ALvD4nHan9VDJlYJwNurVr7H7OV0j2X9)                 | Joint-CTC attention VGGBLSTM trained on CSJ                |

</div></details>

### SE results
<details><summary>expand</summary><div>

We list results from three different models on WSJ0-2mix, which is one the most widely used benchmark dataset for speech separation.

| Model                                             | STOI | SAR   | SDR   | SIR   |
| ------------------------------------------------- | ---- | ----- | ----- | ----- |
| [TF Masking](https://zenodo.org/record/4498554)   | 0.89 | 11.40 | 10.24 | 18.04 |
| [Conv-Tasnet](https://zenodo.org/record/4498562)  | 0.95 | 16.62 | 15.94 | 25.90 |
| [DPRNN-Tasnet](https://zenodo.org/record/4688000) | 0.96 | 18.82 | 18.29 | 28.92 |

</div></details>

### SE demos
<details><summary>expand</summary><div>
You can try the interactive demo with Google Colab. Please click the following button to get access to the demos.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fjRJCh96SoYLZPRxsjF9VDv4Q2VoIckI?usp=sharing)


It is based on ESPnet2. Pretrained models are available for both speech enhancement and speech separation tasks.

</div></details>

### ST results

<details><summary>expand</summary><div>

We list 4-gram BLEU of major ST tasks.

#### end-to-end system
| Task                                              | BLEU  |                                                                                         Pretrained model                                                                                          |
| ------------------------------------------------- | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Fisher-CallHome Spanish fisher_test (Es->En)      | 51.03 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/st1/RESULTS.md#train_spen_lcrm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans) |
| Fisher-CallHome Spanish callhome_evltest (Es->En) | 20.44 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/st1/RESULTS.md#train_spen_lcrm_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans) |
| Libri-trans test (En->Fr)                         | 16.70 |       [link](https://github.com/espnet/espnet/blob/master/egs/libri_trans/st1/RESULTS.md#train_spfr_lc_pytorch_train_pytorch_transformer_bpe_short_long_bpe1000_specaug_asrtrans_mttrans-1)       |
| How2 dev5 (En->Pt)                                | 45.68 |              [link](https://github.com/espnet/espnet/blob/master/egs/how2/st1/RESULTS.md#trainpt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans-1)              |
| Must-C tst-COMMON (En->De)                        | 22.91 |          [link](https://github.com/espnet/espnet/blob/master/egs/must_c/st1/RESULTS.md#train_spen-dede_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans)          |
| Mboshi-French dev (Fr->Mboshi)                    | 6.18  |                                                                                                N/A                                                                                                |

#### cascaded system
| Task                                              | BLEU  | Pretrained model |
| ------------------------------------------------- | :---: | :--------------: |
| Fisher-CallHome Spanish fisher_test (Es->En)      | 42.16 |       N/A        |
| Fisher-CallHome Spanish callhome_evltest (Es->En) | 19.82 |       N/A        |
| Libri-trans test (En->Fr)                         | 16.96 |       N/A        |
| How2 dev5 (En->Pt)                                | 44.90 |       N/A        |
| Must-C tst-COMMON (En->De)                        | 23.65 |       N/A        |

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

| Model                                                                                                        | Notes                                                    |
| :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- |
| [fisher_callhome_spanish.transformer.v1](https://drive.google.com/open?id=1hawp5ZLw4_SIHIT3edglxbKIIkPVe8n3) | Transformer-ST trained on Fisher-CallHome Spanish Es->En |

</div></details>


### MT results

<details><summary>expand</summary><div>

| Task                                              | BLEU  |                                                                        Pretrained model                                                                         |
| ------------------------------------------------- | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Fisher-CallHome Spanish fisher_test (Es->En)      | 61.45 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/mt1/RESULTS.md#trainen_lcrm_lcrm_pytorch_train_pytorch_transformer_bpe_bpe1000) |
| Fisher-CallHome Spanish callhome_evltest (Es->En) | 29.86 | [link](https://github.com/espnet/espnet/blob/master/egs/fisher_callhome_spanish/mt1/RESULTS.md#trainen_lcrm_lcrm_pytorch_train_pytorch_transformer_bpe_bpe1000) |
| Libri-trans test (En->Fr)                         | 18.09 |          [link](https://github.com/espnet/espnet/blob/master/egs/libri_trans/mt1/RESULTS.md#trainfr_lcrm_tc_pytorch_train_pytorch_transformer_bpe1000)          |
| How2 dev5 (En->Pt)                                | 58.61 |              [link](https://github.com/espnet/espnet/blob/master/egs/how2/mt1/RESULTS.md#trainpt_tc_tc_pytorch_train_pytorch_transformer_bpe8000)               |
| Must-C tst-COMMON (En->De)                        | 27.63 |                               [link](https://github.com/espnet/espnet/blob/master/egs/must_c/mt1/RESULTS.md#summary-4-gram-bleu)                                |
| IWSLT'14 test2014 (En->De)                        | 24.70 |                                     [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result)                                      |
| IWSLT'14 test2014 (De->En)                        | 29.22 |                                     [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result)                                      |
| IWSLT'14 test2014 (De->En)                        | 32.2  | [link](https://github.com/espnet/espnet/blob/master/egs2/iwslt14/mt1/README.md)  |
| IWSLT'16 test2014 (En->De)                        | 24.05 |                                     [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result)                                      |
| IWSLT'16 test2014 (De->En)                        | 29.13 |                                     [link](https://github.com/espnet/espnet/blob/master/egs/iwslt16/mt1/RESULTS.md#result)                                      |

</div></details>

### TTS results

<details><summary>ESPnet2</summary><div>

You can listen to the generated samples in the following URL.
- [ESPnet2 TTS generated samples](https://drive.google.com/drive/folders/1H3fnlBbWMEkQUfrHqosKN_ZX_WjO29ma?usp=sharing)

> Note that in the generation we use Griffin-Lim (`wav/`) and [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) (`wav_pwg/`).

You can download pretrained models via `espnet_model_zoo`.
- [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo)
- [Pretrained model list](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv)

You can download pretrained vocoders via `kan-bayashi/ParallelWaveGAN`.
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [Pretrained vocoder list](https://github.com/kan-bayashi/ParallelWaveGAN#results)

</div></details>

<details><summary>ESPnet1</summary><div>

> NOTE: We are moving on ESPnet2-based development for TTS. Please check the latest results in the above ESPnet2 results.

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
- [Single English speaker knowledge distillation-based FastSpeech](https://drive.google.com/open?id=1wG-Y0itVYalxuLAHdkAHO7w1CWFfRPF4)

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

| Model link                                                                                           | Lang  | Fs [Hz] | Mel range [Hz] | FFT / Shift / Win [pt] | Model type                                                              |
| :--------------------------------------------------------------------------------------------------- | :---: | :-----: | :------------: | :--------------------: | :---------------------------------------------------------------------- |
| [ljspeech.wavenet.softmax.ns.v1](https://drive.google.com/open?id=1eA1VcRS9jzFa-DovyTgJLQ_jmwOLIi8L) |  EN   | 22.05k  |      None      |   1024 / 256 / None    | [Softmax WaveNet](https://github.com/kan-bayashi/PytorchWaveNetVocoder) |
| [ljspeech.wavenet.mol.v1](https://drive.google.com/open?id=1sY7gEUg39QaO1szuN62-Llst9TrFno2t)        |  EN   | 22.05k  |      None      |   1024 / 256 / None    | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [ljspeech.parallel_wavegan.v1](https://drive.google.com/open?id=1tv9GKyRT4CDsvUWKwH3s_OfXkiTi0gw7)   |  EN   | 22.05k  |      None      |   1024 / 256 / None    | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [ljspeech.wavenet.mol.v2](https://drive.google.com/open?id=1es2HuKUeKVtEdq6YDtAsLNpqCy4fhIXr)        |  EN   | 22.05k  |    80-7600     |   1024 / 256 / None    | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [ljspeech.parallel_wavegan.v2](https://drive.google.com/open?id=1Grn7X9wD35UcDJ5F7chwdTqTa4U7DeVB)   |  EN   | 22.05k  |    80-7600     |   1024 / 256 / None    | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [ljspeech.melgan.v1](https://drive.google.com/open?id=1ipPWYl8FBNRlBFaKj1-i23eQpW_W_YcR)             |  EN   | 22.05k  |    80-7600     |   1024 / 256 / None    | [MelGAN](https://github.com/kan-bayashi/ParallelWaveGAN)                |
| [ljspeech.melgan.v3](https://drive.google.com/open?id=1_a8faVA5OGCzIcJNw4blQYjfG4oA9VEt)             |  EN   | 22.05k  |    80-7600     |   1024 / 256 / None    | [MelGAN](https://github.com/kan-bayashi/ParallelWaveGAN)                |
| [libritts.wavenet.mol.v1](https://drive.google.com/open?id=1jHUUmQFjWiQGyDd7ZeiCThSjjpbF_B4h)        |  EN   |   24k   |      None      |   1024 / 256 / None    | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [jsut.wavenet.mol.v1](https://drive.google.com/open?id=187xvyNbmJVZ0EZ1XHCdyjZHTXK9EcfkK)            |  JP   |   24k   |    80-7600     |   2048 / 300 / 1200    | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [jsut.parallel_wavegan.v1](https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM)       |  JP   |   24k   |    80-7600     |   2048 / 300 / 1200    | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |
| [csmsc.wavenet.mol.v1](https://drive.google.com/open?id=1PsjFRV5eUP0HHwBaRYya9smKy5ghXKzj)           |  ZH   |   24k   |    80-7600     |   2048 / 300 / 1200    | [MoL WaveNet](https://github.com/r9y9/wavenet_vocoder)                  |
| [csmsc.parallel_wavegan.v1](https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy)      |  ZH   |   24k   |    80-7600     |   2048 / 300 / 1200    | [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)      |

If you want to use the above pretrained vocoders, please exactly match the feature setting with them.

</div></details>

### TTS demo

<details><summary>ESPnet2</summary><div>

You can try the real-time demo in Google Colab.
Please access the notebook from the following button and enjoy the real-time synthesis!

- Real-time TTS demo with ESPnet2  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_tts_realtime_demo.ipynb)

English, Japanese, and Mandarin models are available in the demo.

</div></details>

<details><summary>ESPnet1</summary><div>

> NOTE: We are moving on ESPnet2-based development for TTS. Please check the latest demo in the above ESPnet2 demo.

You can try the real-time demo in Google Colab.
Please access the notebook from the following button and enjoy the real-time synthesis.

- Real-time TTS demo with ESPnet1  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb)

We also provide shell script to perform synthesize.
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
echo "TEXT TO SPEECH IS A TECHNIQUE TO CONVERT TEXT INTO SPEECH." >> example_multi.txt
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

See more details or available models via `--help`.

```sh
synth_wav.sh --help
```

</div></details>

### VC results

<details><summary>expand</summary><div>

- Transformer and Tacotron2 based VC

You can listen to some samples on the [demo webpage](https://unilight.github.io/Publication-Demos/publications/transformer-vc/).

- Cascade ASR+TTS as one of the baseline systems of VCC2020

The [Voice Conversion Challenge 2020](http://www.vc-challenge.org/) (VCC2020) adopts ESPnet to build an end-to-end based baseline system.
In VCC2020, the objective is intra/cross lingual nonparallel VC.
You can download converted samples of the cascade ASR+TTS baseline system [here](https://drive.google.com/drive/folders/1oeZo83GrOgtqxGwF7KagzIrfjr8X59Ue?usp=sharing).

</div></details>

### SLU results

<details><summary>expand</summary><div>


We list the performance on various SLU tasks and dataset using the metric reported in the original dataset paper

| Task                                                              | Dataset                                                              |    Metric     |     Result     |                                                                              Pretrained Model                                         |
| ----------------------------------------------------------------- | :-------------: | :-------------: | :-------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Intent Classification                                                 |     SLURP     |       Acc       |       86.3       |                [link](https://github.com/espnet/espnet/tree/master/egs2/slurp/asr1/README.md)                |
| Intent Classification                                                   |     FSC     |       Acc       |       99.6       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc/asr1/README.md)                |
| Intent Classification                                                  |     FSC Unseen Speaker Set     |       Acc       |       98.6       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_unseen/asr1/README.md)                |
| Intent Classification                                                   |     FSC Unseen Utterance Set     |       Acc       |       86.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_unseen/asr1/README.md)                |
| Intent Classification                                                   |     FSC Challenge Speaker Set     |       Acc       |       97.5       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_challenge/asr1/README.md)                |
| Intent Classification                                                   |     FSC Challenge Utterance Set     |       Acc       |       78.5       |                [link](https://github.com/espnet/espnet/tree/master/egs2/fsc_challenge/asr1/README.md)                |
| Intent Classification                                                   |     SNIPS     |       F1       |       91.7       |                [link](https://github.com/espnet/espnet/tree/master/egs2/snips/asr1/README.md)                |
| Intent Classification                                                   |     Grabo (Nl)   |       Acc       |       97.2       |                [link](https://github.com/espnet/espnet/tree/master/egs2/grabo/asr1/README.md)                |
| Intent Classification                                                   |     CAT SLU MAP (Zn)     |       Acc       |       78.9       |                [link](https://github.com/espnet/espnet/tree/master/egs2/catslu/asr1/README.md)                |
| Intent Classification                                                  |     Google Speech Commands    |       Acc       |       98.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/speechcommands/asr1/README.md)                |
| Slot Filling                                                  |     SLURP     |       SLU-F1       |       71.9       |                [link](https://github.com/espnet/espnet/tree/master/egs2/slurp_entity/asr1/README.md)                |
| Dialogue  Act Classification                                                 |     Switchboard     |       Acc       |       67.5       |                [link](https://github.com/espnet/espnet/tree/master/egs2/swbd_da/asr1/README.md)                |
| Dialogue  Act Classification                                                 |     Jdcinal (Jp)    |       Acc       |       67.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/jdcinal/asr1/README.md)                |
| Emotion Recognition                                                  |     IEMOCAP     |       Acc       |       69.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/iemocap/asr1/README.md)                |
| Emotion Recognition                                                  |     swbd_sentiment     |       Macro F1       |       61.4       |                [link](https://github.com/espnet/espnet/tree/master/egs2/swbd_sentiment/asr1/README.md)                |
| Emotion Recognition                                                  |     slue_voxceleb     |       Macro F1       |       44.0       |                [link](https://github.com/espnet/espnet/tree/master/egs2/slue-voxceleb/asr1/README.md)                |


If you want to check the results of the other recipes, please check `egs2/<name_of_recipe>/asr1/RESULTS.md`.



</div></details>

### CTC Segmentation demo

<details><summary>ESPnet1</summary><div>

[CTC segmentation](https://arxiv.org/abs/2007.09127) determines utterance segments within audio files.
Aligned utterance segments constitute the labels of speech datasets.

As demo, we align start and end of utterances within the audio file `ctc_align_test.wav`, using the example script `utils/asr_align_wav.sh`.
For preparation, set up a data directory:

```sh
cd egs/tedlium2/align1/
# data directory
align_dir=data/demo
mkdir -p ${align_dir}
# wav file
base=ctc_align_test
wav=../../../test_utils/${base}.wav
# recipe files
echo "batchsize: 0" > ${align_dir}/align.yaml

cat << EOF > ${align_dir}/utt_text
${base} THE SALE OF THE HOTELS
${base} IS PART OF HOLIDAY'S STRATEGY
${base} TO SELL OFF ASSETS
${base} AND CONCENTRATE
${base} ON PROPERTY MANAGEMENT
EOF
```

Here, `utt_text` is the file containing the list of utterances.
Choose a pre-trained ASR model that includes a CTC layer to find utterance segments:

```sh
# pre-trained ASR model
model=wsj.transformer_small.v1
mkdir ./conf && cp ../../wsj/asr1/conf/no_preprocess.yaml ./conf

../../../utils/asr_align_wav.sh \
    --models ${model} \
    --align_dir ${align_dir} \
    --align_config ${align_dir}/align.yaml \
    ${wav} ${align_dir}/utt_text
```

Segments are written to `aligned_segments` as a list of file/utterance name, utterance start and end times in seconds and a confidence score.
The confidence score is a probability in log space that indicates how good the utterance was aligned. If needed, remove bad utterances:

```sh
min_confidence_score=-5
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' ${align_dir}/aligned_segments
```

The demo script `utils/ctc_align_wav.sh` uses an already pretrained ASR model (see list above for more models).
It is recommended to use models with RNN-based encoders (such as BLSTMP) for aligning large audio files;
rather than using Transformer models that have a high memory consumption on longer audio data.
The sample rate of the audio must be consistent with that of the data used in training; adjust with `sox` if needed.
A full example recipe is in `egs/tedlium2/align1/`.

</div></details>

<details><summary>ESPnet2</summary><div>

[CTC segmentation](https://arxiv.org/abs/2007.09127) determines utterance segments within audio files.
Aligned utterance segments constitute the labels of speech datasets.

As demo, we align start and end of utterances within the audio file `ctc_align_test.wav`.
This can be done either directly from the Python command line or using the script `espnet2/bin/asr_align.py`.

From the Python command line interface:

```python
# load a model with character tokens
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader(cachedir="./modelcache")
wsjmodel = d.download_and_unpack("kamo-naoyuki/wsj")
# load the example file included in the ESPnet repository
import soundfile
speech, rate = soundfile.read("./test_utils/ctc_align_test.wav")
# CTC segmentation
from espnet2.bin.asr_align import CTCSegmentation
aligner = CTCSegmentation( **wsjmodel , fs=rate )
text = """
utt1 THE SALE OF THE HOTELS
utt2 IS PART OF HOLIDAY'S STRATEGY
utt3 TO SELL OFF ASSETS
utt4 AND CONCENTRATE ON PROPERTY MANAGEMENT
"""
segments = aligner(speech, text)
print(segments)
# utt1 utt 0.26 1.73 -0.0154 THE SALE OF THE HOTELS
# utt2 utt 1.73 3.19 -0.7674 IS PART OF HOLIDAY'S STRATEGY
# utt3 utt 3.19 4.20 -0.7433 TO SELL OFF ASSETS
# utt4 utt 4.20 6.10 -0.4899 AND CONCENTRATE ON PROPERTY MANAGEMENT
```

Aligning also works with fragments of the text.
For this, set the `gratis_blank` option that allows skipping unrelated audio sections without penalty.
It's also possible to omit the utterance names at the beginning of each line, by setting `kaldi_style_text` to False.

```python
aligner.set_config( gratis_blank=True, kaldi_style_text=False )
text = ["SALE OF THE HOTELS", "PROPERTY MANAGEMENT"]
segments = aligner(speech, text)
print(segments)
# utt_0000 utt 0.37 1.72 -2.0651 SALE OF THE HOTELS
# utt_0001 utt 4.70 6.10 -5.0566 PROPERTY MANAGEMENT
```

The script `espnet2/bin/asr_align.py` uses a similar interface. To align utterances:

```sh
# ASR model and config files from pretrained model (e.g. from cachedir):
asr_config=<path-to-model>/config.yaml
asr_model=<path-to-model>/valid.*best.pth
# prepare the text file
wav="test_utils/ctc_align_test.wav"
text="test_utils/ctc_align_text.txt"
cat << EOF > ${text}
utt1 THE SALE OF THE HOTELS
utt2 IS PART OF HOLIDAY'S STRATEGY
utt3 TO SELL OFF ASSETS
utt4 AND CONCENTRATE
utt5 ON PROPERTY MANAGEMENT
EOF
# obtain alignments:
python espnet2/bin/asr_align.py --asr_train_config ${asr_config} --asr_model_file ${asr_model} --audio ${wav} --text ${text}
# utt1 ctc_align_test 0.26 1.73 -0.0154 THE SALE OF THE HOTELS
# utt2 ctc_align_test 1.73 3.19 -0.7674 IS PART OF HOLIDAY'S STRATEGY
# utt3 ctc_align_test 3.19 4.20 -0.7433 TO SELL OFF ASSETS
# utt4 ctc_align_test 4.20 4.97 -0.6017 AND CONCENTRATE
# utt5 ctc_align_test 4.97 6.10 -0.3477 ON PROPERTY MANAGEMENT
```

The output of the script can be redirected to a `segments` file by adding the argument `--output segments`.
Each line contains file/utterance name, utterance start and end times in seconds and a confidence score; optionally also the utterance text.
The confidence score is a probability in log space that indicates how good the utterance was aligned. If needed, remove bad utterances:

```sh
min_confidence_score=-7
# here, we assume that the output was written to the file `segments`
awk -v ms=${min_confidence_score} '{ if ($5 > ms) {print} }' segments
```

See the module documentation for more information.
It is recommended to use models with RNN-based encoders (such as BLSTMP) for aligning large audio files;
rather than using Transformer models that have a high memory consumption on longer audio data.
The sample rate of the audio must be consistent with that of the data used in training; adjust with `sox` if needed.

Also, we can use this tool to provide token-level segmentation information if we prepare a list of tokens instead of that of utterances in the `text` file. See the discussion in https://github.com/espnet/espnet/issues/4278#issuecomment-1100756463.

</div></details>

## Citations

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
@inproceedings{li2020espnet,
  title={{ESPnet-SE}: End-to-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph Boeddeker and Zhuo Chen and Shinji Watanabe},
  booktitle={Proceedings of IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE},
}
@article{arora2021espnet,
  title={ESPnet-SLU: Advancing Spoken Language Understanding through ESPnet},
  author={Arora, Siddhant and Dalmia, Siddharth and Denisov, Pavel and Chang, Xuankai and Ueda, Yushi and Peng, Yifan and Zhang, Yuekai and Kumar, Sujay and Ganesan, Karthik and Yan, Brian and others},
  journal={arXiv preprint arXiv:2111.14706},
  year={2021}
}
```
