# ESPnet2 ASR2 Recipe TEMPLATE

This is a template of ASR2 recipe for ESPnet2.
The difference from ASR1 is that discrete tokens are used as input instead of conventional audios / spectrum features.

## Table of Contents

* [ESPnet2 ASR2 Recipe TEMPLATE](#espnet2-asr2-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Speed perturbation](#2-speed-perturbation)
    * [3\. Wav format](#3-wav-format)
    * [4\. Generate discrete tokens](#4-generate-discrete-tokens)
    * [5\. Generate dump folder](#5-generate-dump-folder)
    * [6\. Removal of long / short data](#6-removal-of-long--short-data)
    * [7\. Input / Output Token list generation](#7-input-output-token-list-generation)
    * [8\. LM statistics collection](#8-lm-statistics-collection)
    * [9\. LM training](#9-lm-training)
    * [10\. LM perplexity](#10-lm-perplexity)
    * [11\. Ngram-LM training](#11-ngram-lm-training)
    * [12\. ASR statistics collection](#12-asr-statistics-collection)
    * [13\. ASR training](#13-asr-training)
    * [14\. ASR inference](#14-asr-inference)
    * [15\. ASR scoring](#15-asr-scoring)
    * [16\-18\. (Optional) Pack results for upload](#16-18-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [LibriSpeech training](#librispeech-training)
  * [How to create asr2 recipes from asr1](#how-to-create)
  * [Tips for asr2 experiments](#asr2-tips)
  * [Related works](#related-works)

## Recipe flow

ASR2 recipe consists of 15 stages.

### 1. Data preparation

Data preparation stage.

#### ESPnet format:

It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets. It's the same as `asr1` tasks.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Speed perturbation

Augment training data with speed perturbation. `data/${train_set}_spXX` would be generated (`XX` means the speed factor). This step is optional.

### 3. Wav format

Format the wave files in `wav.scp` to a single format (wav / flac / kaldi_ark).

### 4. Generate discrete tokens

The discrete tokens of the input speech signals are generated. For ASR2 task, the input is the discrete tokens (from self-supervised learning (SSL) features) and the target is the ASR transcriptions. After getting the discrete tokens (usually in integers), they will be converted to CJK characters, which are more convenient in tokenization.
#### Input / Target / Process of data preparation

- Stages:
  1. Generate SSL features for train / valid / test sets.
  2. Train the K-Means model on a subset from training data.
  3. Generate K-Means-based discrete tokens for train / valid / test sets.
  4. (Optional) Measure the discrete tokens quality if forced-alignment is accessible.


### 5. Generate dump folder

Dumping stage.
This stage move necessary files for training from `data` folder to `dump` folder.

### 6. Removal of long / short data

TODO.

### 7. Token list generation

Token list (BPE / Char / etc) generation for both input and targets.

### 8. LM statistics collection

Neural-network (NN) based Language model (LM) is optional for ASR task. You can skip stage 5-8 by setting `--use_lm false`.
Statistics calculation stage.
It collects the shape information of LM texts and calculates statistics for LM training.

### 9. LM training

NN-based LM model training stage.
You can change the training setting via `--lm_config` and `--lm_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 10. LM perplexity

NN-based LM evaluation stage. Perplexity (PPL) is computed against the trained model

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 11. N-gram LM training

N-gram-based LM model training stage.


### 12. ASR statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for ASR training.

### 13. ASR training

ASR model training stage.
You can change the training setting via `--asr_config` and `--asr_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 14. ASR inference

ASR inference stage.

### 15. ASR scoring

ASR scoring stage: error rates (char / word / token) are computed.

### 16-18. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)

#### Stage 16-18: Upload model

Upload the trained model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## How to create asr2 recipes from asr1
```
1. mkdir egs2/<recipe name>/asr2
2. cd egs2/<recipe name>/asr2
3. ../../TEMPLATE/asr2/setup.sh . # create the default files
4. cd local
5. ln -s ../../asr1/local/* .     # add symlinks for data preparation files in asr1 except for path.sh
6. cd ../
7. cp ../../librispeech/asr2/run.sh .         # copy run.sh
8. cp ../../librispeech/asr2/conf/tuning/train_discrete_asr_e_branchformer1.yaml conf/ # copy training conf
9. cp ../../librispeech/asr2/conf/decode_ctc0.3.yaml conf/     # copy confs
10. EDIT run.sh by checking ../asr1/run.sh
  a. We may skip an LM by adding an option `--use_lm false`
  b. change language-relatedd setting (e.g., tgt_lang, tgt_nbpe (e.g., change to char for CKJ))
  c. training, dev, and test sets
  d. config files
```

## Tips for asr2 experiments
* **Kmeans is important**
   * SSL model choice can affect the performance a lot, e.g. wavlm models may not work for non-English data,
   * Layer selection is also important: different layers retain different information. For example, based the training criterion, the 24-th layer of HuBERT_large is trying to match the information from HuBERT_base layer 9. If you didn't have experience, please refer to the Fig. 4 of this [CCA paper](https://arxiv.org/pdf/2211.03929.pdf), which is usually helpful.
   * Number of kmeans clusters also affect the variance in pronunciation, etc.
   * Please check the kmeans labels in `dump/extracted/{kmeans_feat_type}/layer{layer}/{dset}/pseudo_label_km{ncluseters}.txt`. In my experience, a good km result for ASR should have an obvious pattern of repeatitions, e.g. mhubert_base model on covost2  (Spanish speech)
      ```
      00252a1e8f956757a1fef398e3fa3c659425c690b493b2517498a4df162536d629dad2372d963e9381c81c3567b04256896841d533f581ae8a6d55dfdc141306-common_voice_es_19679587 210 210 937 565 713 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 453 115 277 277 521 798 306 306 937 417 306 306 306 306 306 306 306 306 306 713 417 306 306 958 306 306 306 53 53 53 942 942 942 713 942 942 942 328 700 700 623 623 938 938 333 983 983 597 597 709 709 23 238 238 557 148 832 832 784 784 784 784 784 916 93 233 948 62 220 220 220 220 534 302 368 661 280 529 24 24 435 21 540 540 997 997 87 87 50 994 211 671 638 362 788 788 520 230 11 268 575 5 5 5 717 919 478 721 566 922 654 654 654 654 625 756 756 76 76 250 17 17 410 219 599 599 599 599 304 304 570 248 12 409 385 160 547 175 20 824 581 213 773 975 875 733 431 431 508 508 648 88 660 660 660 896 896 544 410 410 399 373 869 103 758 867 33 319 168 852 852 555 555 207 454 454 33 93 233 948 948 62 269 442 442 302 302 493 280 84 758 758 758 867 33 874 750 750 750 473 681 609 173 173 683 959 959 970 352 352 352 129 8 876 222 381 181 694 967 919 919 840 840 260 260 260 129 598 598 613 613 613 965 965 965 743 743 620 141 60 60 714 67 67 67 359 359 359 208 546 307 359 307 307 307 307 277 277 277 453 115 307 713 48 277 115 115 115 423 423 423 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 798 16 277 277 277 995 19 19 19 19 19 19 19 19 19 19 19 19 19 19 452 19 19 452 452 452 452 452 452 452 452 452 452 452 452 452 452 322 322'
      ```
      The following is a bad example, wavlm_large on covost2 (Spanish speech)
      ```
      00252a1e8f956757a1fef398e3fa3c659425c690b493b2517498a4df162536d629dad2372d963e9381c81c3567b04256896841d533f581ae8a6d55dfdc141306-common_voice_es_19679587 109 12 429 682 682 886 288 143 97 331 97 143 97 228 143 919 628 919 628 191 790 552 790 810 790 810 790 435 810 435 378 432 432 949 435 435 949 949 188 188 188 188 790 188 188 22 22 952 188 188 22 710 51 22 783 188 331 278 51 810 278 142 59 278 437 754 952 14 764 764 437 351 985 204 284 467 503 338 755 755 855 855 849 749 344 344 669 705 705 642 927 927 927 889 45 987 597 182 36 787 212 410 967 183 183 840 428 428 259 295 517 452 452 806 644 644 776 67 364 648 423 27 417 174 601 765 484 186 943 468 738 354 209 44 912 930 930 930 442 442 651 7 7 7 383 139 218 311 311 771 814 166 166 345 471 608 377 377 245 292 575 504 730 58 656 627 75 580 762 972 120 146 146 931 716 686 776 67 491 828 828 881 741 312 879 879 280 280 146 146 73 709 729 11 469 11 966 945 433 153 582 123 900 619 100 100 100 889 987 153 167 366 451 359 169 169 913 121 482 482 482 11 664 224 89 550 153 858 124 858 212 789 667 523 523 932 25 409 386 61 992 482 482 759 924 620 516 347 805 901 317 317 518 852 852 66 66 781 904 807 356 368 368 409 409 63 63 20 863 998 13 303 13 13 385 811 458 811 71 71 439 756 821 71 439 378 370 835 370 439 130 439 475 475 439 130 939 34 939 949 835 835 949 835 949 835 435 949 435 435 435 435 949 435 435 810 435 949 949 331 435 435 435 949 949 390 435 737 358 378 378 949 949 949 390 949 949 234 949 234 234 65 90 234 350 390 90 390 350 350 350 350 43 350 331 43 126 126 131 804 572 376 159 862 862
      ```

* Subword modeling (BPE) vocab_size is also important, for both source and target data. Very large vocab size results in data sparseness.
* Try to monitor the sequence length, I list some commands used here:
  * Original wavform length
    ```
    <dump/audio_raw/{train_set}/utt2num_samples awk '{total+=$2}END{print(total/NR)}'
    ```
  * Pseudo-label length
    ```
    <dump/extracted/{ssl_feat_type}/layer{layer}/{train_set}/pseudo_labels_km{nclusters}.txt awk '{total+=NF-1}END{print(total/NR)}'
    ```
  * Discrete token length after de-duplication
    ```
    <dump/raw/${train_set}/text.rm.${src_lang} awk '{total+=length($2)}END{print(total/NR)}'
    ```
  * Discrete token length after BPE
    ```
    <exp/{asr_stats_dir}/train/src_text_shape.bpe awk '{split($2, lst, ","); total+=lst[1]}END{print(total/NR)}'
    ```
  * ASR transcription length after BPE
    ```
    <exp/{asr_stats_dir}/train/text_shape.bpe awk '{split($2, lst, ","); total+=lst[1]}END{print(total/NR)}'
    ```
* During training, please observe the accuracy on the valid set. Usually it would reach >80% within 10 epochs. But it also depends on the data and optimization hyper-parameters.
```

## Related works
```
@INPROCEEDINGS{9054224,
  author={Baevski, Alexei and Mohamed, Abdelrahman},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Effectiveness of Self-Supervised Pre-Training for ASR},
  year={2020},
  volume={},
  number={},
  pages={7694-7698},
  doi={10.1109/ICASSP40776.2020.9054224}}

@article{chang2023exploration,
  title={Exploration of Efficient End-to-End ASR using Discretized Input from Self-Supervised Learning},
  author={Chang, Xuankai and Yan, Brian and Fujita, Yuya and Maekaku, Takashi and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2305.18108},
  year={2023}
}
```
