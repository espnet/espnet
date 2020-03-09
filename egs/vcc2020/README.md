# Recipe for Voice Conversion Challenge 2020 baseline : Cascade ASR + TTS

Official homepage: [http://www.vc-challenge.org/](http://www.vc-challenge.org/)

## Introduction

This recipe describes baseline for the Voice Conversion (VC) Challenge 2020 (VCC2020 for short). VCC2020 contains two tasks. In either tasks, the source speech is always an English speech from a native speaker. Task 1 requires to convert to a English target speaker with a small amount of English parallel training set, and task 2 requires to convert to a non-English (German/Finnish/Mandarin) target speaker with a non-English training set. Task 2 is also referred as cross-lingual VC.

## Method

A naive approach for VC is a cascade of an automatic speech recognition (ASR) model and a text-to-speech (TTS) model. In this recipe we revisit this method under an end-to-end (E2E) framework.

- Specifically, we train a (speaker independent) ASR model, and a separate speaker dependent TTS model for each target speaker. First, the ASR model takes the source English speech as input and output the transcribed text. Then, the TTS model of the target speaker synthesizes the converted speech using the recognition result.
- Since the size of training set of each target speaker is to limited for E2E TTS learning, we use a pretraining-finetuning scheme to improve the performance.
- Neural waveform generation module (vocoder) has brought significant improvement to VC in recent years. In this baseline we adopt the open source [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) (PWG), since it enables high-quality, faster than real-time acoustic feature to waveform decoding.

The training flow is as following:

1. ASR training. A [Transformer-based ASR model](https://github.com/espnet/espnet/tree/master/egs/librispeech/asr1) is used. ESPnet provides a [LibriSpeech pretrained model](https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--large-lstm-lm).
2. TTS pretraining. We use a [multi-speaker, x-vector Transformer-TTS model](https://github.com/espnet/espnet/tree/master/egs/libritts/tts1). In task 1, we use the [Libritts pretrained model](https://github.com/espnet/espnet/blob/master/egs/libritts/tts1/RESULTS.md#v050-first-multi-speaker-transformer-1024-pt-window--256-pt-shift--x-vector-with-add-integration--fast-gl-64-iters) provided by ESPnet. In task 2, corpora of two languages are used for pretraining: English and the language of the target speaker.
3. TTS finetuning. We update all parameters using the training set of the target speaker.
4. PWG training. We pool the training data from all available speakers in each tasks.

## Recipe structure

- `tts1_en_[de/fi/zh]`: TTS pretraining using English and the non-English language (de:German; fi:Finnish; zh:Mandarin).
- `voc1`: PWG training.
- `vc1_task[1/2]`: TTS finetuning, and the conversion phase for task 1 and task 2.

## Datasets and preparation.

The following datasets are used to train this baseline method.

- LibriSpeech: Contains English data for ASR training. Can be downloaded automatically in the recipe.
- LibriTTS: Contains English data for task1 TTS pretraining. Can be downloaded automatically in the recipe.
- M-AILABS: Contains English and German data for task2 TTS pretraining. Can be downloaded automatically in the recipe.
- [CSS10](https://www.kaggle.com/bryanpark/finnish-single-speaker-speech-dataset): Contains Finnish data for task2 TTS pretraining. To download this dataset, Kaggle membership is requires. Please download and put in the desired directory. (default is `tts_en_fi/downloads/`)
- CSMSC: Contains Mandarin data for task2 TTS pretraining. Can be downloaded automatically in the recipe.
- VCC2020: Contains the training data of the challenge. Please follow the instruction from the organizers and put in the desired directory. (default is `vc1/downloads/`)

## Usage

Please see the readme in each recipe.

## Author

Wen-Chin Huang @ Nagoya University (2020/03)  
If you have any questions, please open an issue.  
Or contact trhough email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp
