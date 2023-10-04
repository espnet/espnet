# ESPnet2 S2T1 Recipe TEMPLATE

This is a template of S2T1 recipe for ESPnet2. It is based on ASR1, but follows the style of OpenAI's Whisper to train a single encoder-decoder model for various speech processing tasks. 
Specifically, it uses special tokens as task specifiers (e.g., transcribe, translate) or prediction targets (e.g., language ID) so that a single model can perform multiple tasks for multiple languages. It further supports conditional generation where the condition is the previous sentence within the long talk.

More details can be found in our [OWSM](https://arxiv.org/abs/2309.13876) paper (ASRU 2023).


## Table of Contents

* [ESPnet2 S2T1 Recipe TEMPLATE](#espnet2-s2t1-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Speed perturbation](#2-speed-perturbation)
    * [3\. Wav format](#3-wav-format)
    * [4\. Remove long or short data](#4-remove-long-or-short-data)
    * [5\. Generate token list](#5-generate-token-list)
    * [6\. LM statistics collection](#6-lm-statistics-collection)
    * [7\. LM training](#7-lm-training)
    * [8\. LM perplexity](#8-lm-perplexity)
    * [9\. Ngram-LM training](#9-ngram-lm-training)
    * [10\. S2T statistics collection](#10-s2t-statistics-collection)
    * [11\. S2T training](#11-s2t-training)
    * [12\. S2T inference](#12-s2t-inference)
    * [13\. S2T scoring](#13-s2t-scoring)
    * [14\-16\. (Optional) Pack results for upload](#14-16-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [OWSM Training](#owsm-training)
  * [Related works](#related-works)

## Recipe flow

S2T1 recipe consists of 16 stages.

### 1. Data preparation

Data preparation stage.

#### ESPnet format:

It calls `local/data.sh` to creates [Kaldi-style data](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory) directories in `data/` for training and validation sets.

The training data has the following format:
```
<sop> prev<sos><category><task><starttime1> utt1<endtime1><starttime2> utt2<endtime2><eos>
```
where `<sop>` is a special token denoting the start of prev/prompt sentence. The timestamps are also treated as special tokens because the audio has a fixed length (30s) and resolution (20ms or 40ms). An example looks like:

```
<sop> I'm going to talk today about energy and climate.<sos><en><transcribe><0.00> And that might seem a bit surprising, because my full-time work at the foundation is mostly about vaccines and seeds, about the things that we need to invent and deliver to help the poorest two billion live better lives.<14.12><15.36> But energy and climate are extremely important to these people; in fact, more important than to anyone else on the planet.<24.26><eos>
```

During data preparation, three text files are generated:
- `text` contains the normal target sentence, i.e., the text between `<sos>` and `<eos>`.
- `text.prev` contains the previous sentence, i.e., the text between `<sop>` and `<sos>`. This might be unavailable at the beginning of a talk. In such cases, a special token `<na>` will be used.
- `text.ctc` contains the ASR transcript without any special token, which is used for the CTC loss. For ASR utterances, this can be derived from `text`, but for ST utterances, this is in a different language. If the ASR transcription is not available, `<na>` will be used.


### 2. Speed perturbation

Augment training data with speed perturbation. `data/${train_set}_spXX` would be generated (`XX` means the speed factor). This step is optional. Note that the timestamps need to be changed as well.

### 3. Wav format

Format the wave files in `wav.scp` to a single format (wav / flac / kaldi_ark).

### 4. Remove long or short data

Remove too long or too short data.

### 5. Generate token list

Generate token list from the training data. BPE tokens are used.

### 6. LM statistics collection

Neural-network (NN) based Language model (LM) is optional for S2T1 task. You can skip stage 6-9 by setting `--use_lm false`.
Statistics calculation stage.
It collects the shape information of LM texts and calculates statistics for LM training.

### 7. LM training

NN-based LM model training stage.
You can change the training setting via `--lm_config` and `--lm_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 8. LM perplexity

NN-based LM evaluation stage. Perplexity (PPL) is computed against the trained model

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 9. Ngram LM training

N-gram-based LM model training stage.


### 10. S2T statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for S2T training.

### 11. S2T training

S2T model training stage.
You can change the training setting via `--s2t_config` and `--s2t_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 12. S2T inference

S2T inference stage. We can perform ASR or ST using any prepared test data.

### 13. S2T scoring

Calculate ASR error rates (char / word / token).

### 14-16. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)
- Upload the trained model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## How to run

### OWSM training

We have created several recipes for [OWSM](https://arxiv.org/abs/2309.13876) training. Please check `egs2/mixed_v1`, `egs2/mixed_v2`, `egs2/mixed_v3` for more information.

## Related work
```
@article{peng2023reproducing,
  title={Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data},
  author={Peng, Yifan and Tian, Jinchuan and Yan, Brian and Berrebbi, Dan and Chang, Xuankai and Li, Xinjian and Shi, Jiatong and Arora, Siddhant and Chen, William and Sharma, Roshan and others},
  journal={arXiv preprint arXiv:2309.13876},
  year={2023}
}
```
