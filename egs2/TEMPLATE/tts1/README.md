# ESPnet2 TTS Recipe TEMPLATE

This is a template of TTS recipe for ESPnet2.

## Table of Contents

* [ESPnet2 TTS Recipe TEMPLATE](#espnet2-tts-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Wav dump or Feature extraction](#2-wav-dump-or-feature-extraction)
    * [3\. Removal of long / short data](#3-removal-of-long--short-data)
    * [4\. Token list generation](#4-token-list-generation)
    * [5\. TTS statistics collection](#5-tts-statistics-collection)
    * [6\. TTS training](#6-tts-training)
    * [7\. TTS decoding](#7-tts-decoding)
    * [8\-9\. (Optional) Pack results for upload](#8-9-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [FastSpeech training](#fastspeech-training)
    * [FastSpeech2 training](#fastspeech2-training)
    * [Multi speaker model with X-vector training](#multi-speaker-model-with-x-vector-training)
  * [Supported text frontend](#supported-text-frontend)
  * [Supported text cleaner](#supported-text-cleaner)
  * [Supported Models](#supported-models)
    * [Single speaker model](#single-speaker-model)
    * [Multi speaker model](#multi-speaker-model)
  * [FAQ](#faq)
    * [ESPnet1 model is compatible with ESPnet2?](#espnet1-model-is-compatible-with-espnet2)
    * [How to change minibatch size in training?](#how-to-change-minibatch-size-in-training)
    * [How to make a new recipe for my own dataset?](#how-to-make-a-new-recipe-for-my-own-dataset)
    * [How to add a new g2p module?](#how-to-add-a-new-g2p-module)
    * [How to add a new cleaner module?](#how-to-add-a-new-cleaner-module)
    * [How to use trained model in python?](#how-to-use-trained-model-in-python)
    * [How to get pretrained models?](#how-to-get-pretrained-models)
    * [How to load the pretrained model?](#how-to-load-the-pretrained-model)
    * [How to finetune the pretrained model?](#how-to-finetune-the-pretrained-model)
    * [How to add a new model?](#how-to-add-a-new-model)
    * [How to test my model with an arbitrary given text?](#how-to-test-my-model-with-an-arbitrary-given-text)
    * [How to handle the errors in validate_data_dir.sh?](#how-to-handle-the-errors-in-validate_data_dirsh)
    * [Why the model generate meaningless speech at the end?](#why-the-model-generate-meaningless-speech-at-the-end)
    * [Why the model cannot be trained well with my own dataset?](#why-the-model-cannot-be-trained-well-with-my-own-dataset)
    * [How is the duration for FastSpeech2 generated?](#how-is-the-duration-for-fastspeech2-generated)
    * [Why the output of Tacotron2 or Transformer is non-deterministic?](#why-the-output-of-tacotron2-or-transformer-is-non-deterministic)

## Recipe flow

TTS recipe consists of 9 stages.

### 1. Data preparation

Data preparation stage.
It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Wav dump or Feature extraction

Feature extraction stage.
The processing in this stage is changed according to `--feats_type` option (Default: `feats_type=raw`).
In the case of `feats_type=raw`, reformat `wav.scp` in date directories.
In the other cases (`feats_type=fbank` and `feats_type=stft`), feature extraction with Librosa will be performed.
Since the performance is almost the same, we recommend using `feats_type=raw`.

Additionaly, we support X-vector extraction in this stage as you can use in ESPnet1.
If you specify `--use_xvector true` (Default: `use_xvector=false`), we extract mfcc features, vad decision, and X-vector.
This processing requires the compiled kaldi, please be careful.

### 3. Removal of long / short data

Processing stage to remove long and short utterances from the training and validation data.
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

### 4. Token list generation

Token list generation stage.
It generates token list (dictionary) from `srctexts`.
You can change the tokenization type via `--token_type` option.
`token_type=char` and `token_type=phn` are supported.
If `--cleaner` option is specified, the input text will be cleaned with the specified cleaner.
If `token_type=phn`, the input text will be converted with G2P module specified by `--g2p` option.

See also:
- [Supported text cleaner](#supported-text-cleaner).
- [Supported text frontend](#supported-text-frontend).

### 5. TTS statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training data).

### 6. TTS training

TTS model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 7. TTS decoding

TTS model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 8-9. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)

## How to run

Here, we show the procedure to run the recipe using `egs2/ljspeech/tts1`.

Move on the recipe directory.
```sh
$ cd egs2/ljspeech/tts1
```

Modify `LJSPEECH` variable in `db.sh` if you want to change the download directory.
```sh
$ vim db.sh
```

Modify `cmd.sh` and `conf/*.conf` if you want to use job scheduler.
See the detail in [using job scheduling system](https://espnet.github.io/espnet/parallelization.html).
```sh
$ vim cmd.sh
```

Run `run.sh`, which conducts all of the stages explained above.
```sh
$ ./run.sh
```
As a default, we train Tacotron2 (`conf/train.yaml`) with `feats_type=raw` + `token_type=phn`.

Then, you can get the following directories in the recipe directory.
```sh
├── data/ # Kaldi-style data directory
│   ├── dev/        # validation set
│   ├── eval1/      # evaluation set
│   ├── token_list/ # token list (dictionary)
│   └── tr_no_dev/  # training set
├── dump/ # feature dump directory
│   └── raw/
│       ├── dev/       # validation set
│       ├── eval1/     # evaluation set
│       ├── srctexts   # text to create token list
│       └── tr_no_dev/ # training set
└── exp/ # experiment directory
    ├── tts_stats_raw_phn_tacotron_g2p_en_no_space # statistics
    └── tts_train_raw_phn_tacotron_g2p_en_no_space # model
        ├── att_ws/                # attention plot during training
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── decode_train.loss.ave/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval1/ # evaluation set
        │        ├── att_ws/      # attention plot in decoding
        │        ├── probs/       # stop probability plot in decoding
        │        ├── norm/        # generated features
        │        ├── denorm/      # generated denormalized features
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── log/         # log directory
        │        ├── durations    # duration of each input tokens
        │        ├── feats_type   # feature type
        │        ├── focus_rates  # focus rate
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_5best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```
In decoding, we use Griffin-Lim for waveform generation.
If you want to combine with neural vocoder, please use [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN).
See the detail in [decoding with ESPnet-TTS model's feature](https://github.com/kan-bayashi/ParallelWaveGAN#decoding-with-espnet-tts-models-features).

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop-stage` options.
```sh
$ ./run.sh --stage 1 --stop-stage 1
$ ./run.sh --stage 2 --stop-stage 2
$ ./run.sh --stage 3 --stop-stage 3
```
This might helps you to understand each stage's processing and directory structure.

### FastSpeech training

If you want to train FastSpeech, additional steps with the teacher model are needed.
Please make sure you already finished the training of the teacher model (Tacotron2 or Transformer-TTS).

First, decode all of data including training, validation, and evaluation set.
```sh
# specify teacher model directory via --tts_exp option
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --test_sets "tr_no_dev dev eval1"
```
This will generate `durations` for training, validation, and evaluation sets in `exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.ave`.

Then, you can train FastSpeech by specifying the directory including `durations` via `--teacher_dumpdir` option.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.ave
```

In the above example, we use generated mel-spectrogram as the target, which is known as knowledge distillation training.
If you want to use groundtruth mel-spectrogram as the target, we need to use teacher forcing in decoding.
```sh
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```
You can get the groundtruth aligned durations in `exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave`.

Then, you can train FastSpeech without knowledge distillation.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave
```

### FastSpeech2 training

The procedure is almost the same as FastSpeech but we **MUST** use teacher forcing in decoding.
```sh
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```

To train FastSpeech2, we use additional feature (F0 and energy).
Therefore, we need to start from `stage 5` to calculate additional statistics.
```sh
$ ./run.sh --stage 5 \
    --train_config conf/tuning/train_fastspeech2.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --write_collected_feats true
```
where `--tts_stats_dir` is the option to specify the directory to dump Statistics, and `--write_collected_feats` is the option to dump features in statistics calculation.
The use of `--write_collected_feats` is optional but it helps to accelerate the training.

### Multi-speaker model with X-vector training

First, you need to run from the stage 2 and 3 with `--use_xvector true` to extract X-vector.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_xvector true
```
You can find the extracted X-vector in `dump/xvector/*/xvector.{ark,scp}`.
Then, you can run the training with the config which has `spk_embed_dim: 512` in `tts_conf`.
```yaml
# e.g.
tts_conf:
    spk_embed_dim: 512               # dimension of speaker embedding
    spk_embed_integration_type: add  # how to integrate speaker embedding
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_xvector true --train_config /path/to/your_xvector_config.yaml
```

You can find the example config in [`egs2/vctk/tts1/conf/tuning`](../../vctk/tts1/conf/tuning).

## Supported text frontend

You can change via `--g2p` option in `tts.sh`.

- `none`: Just separate by space
    - e.g.: `HH AH0 L OW1 <space> W ER1 L D` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`
- `g2p_en`: [Kyubyong/g2p](https://github.com/Kyubyong/g2p)
    - e.g. `Hello World` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`
- `g2p_en_no_space`: [Kyubyong/g2p](https://github.com/Kyubyong/g2p)
    - Same G2P but do not use word separator
    - e.g. `Hello World` -> `[HH, AH0, L, OW1, W, ER1, L, D]`
- `pyopenjtalk`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - e.g. `こ、こんにちは` -> `[k, o, pau, k, o, N, n, i, ch, i, w, a]`
- `pyopenjtalk_kana`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Use kana instead of phoneme
    - e.g. `こ、こんにちは` -> `[コ, 、, コ, ン, ニ, チ, ワ]`
- `pyopenjtalk_accent`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Add accent labels in addition to phoneme labels
    - e.g. `こ、こんにちは` -> `[k, 1, 0, o, 1, 0, k, 5, -4, o, 5, -4, N, 5, -3, n, 5, -2, i, 5, -2, ch, 5, -1, i, 5, -1, w, 5, 0, a, 5, 0]`
- `pyopenjtalk_accent_with_pause`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Add a pause label in addition to phoneme and accenet labels
    - e.g. `こ、こんにちは` -> `[k, 1, 0, o, 1, 0, pau, k, 5, -4, o, 5, -4, N, 5, -3, n, 5, -2, i, 5, -2, ch, 5, -1, i, 5, -1, w, 5, 0, a, 5, 0]`
- `pypinyin`: [mozillanzg/python-pinyin](https://github.com/mozillazg/python-pinyin)
    - e.g. `卡尔普陪外孙玩滑梯。` -> `[ka3, er3, pu3, pei2, wai4, sun1, wan2, hua2, ti1, 。]`
- `pypinyin_phone`: [mozillanzg/python-pinyin](https://github.com/mozillazg/python-pinyin)
    - Separate into first and last parts
    - e.g. `卡尔普陪外孙玩滑梯。` -> `[k, a3, er3, p, u3, p, ei2, wai4, s, un1, uan2, h, ua2, t, i1, 。]`
- `espeak_ng_arabic`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `السلام عليكم` -> `[ʔ, a, s, s, ˈa, l, aː, m, ʕ, l, ˈiː, k, m]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)

You can see the code example from [here](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/test/espnet2/text/test_phoneme_tokenizer.py).

## Supported text cleaner

You can change via `--cleaner` option in `tts.sh`.

- `none`: No text cleaner.
- `tacotron`: [keithito/tacotron](https://github.com/keithito/tacotron)
    - e.g.`"(Hello-World);  & jr. & dr."` ->`HELLO WORLD, AND JUNIOR AND DOCTOR`
- `jaconv`: [kazuhikoarase/jaconv](https://github.com/kazuhikoarase/jaconv)
    - e.g. `”あらゆる”　現実を　〜　’すべて’ 自分の　ほうへ　ねじ曲げたのだ。"` -> `"あらゆる" 現実を ー \'すべて\' 自分の ほうへ ねじ曲げたのだ。`

You can see the code example from [here](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/test/espnet2/text/test_cleaner.py).

## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `tts.sh`.

### Single speaker model

- [Tacotron 2](https://arxiv.org/abs/1712.05884)
- [Transformer-TTS](https://arxiv.org/abs/1809.08895)
- [FastSpeech](https://arxiv.org/abs/1905.09263)
- [FastSpeech2](https://arxiv.org/abs/2006.04558) ([FastPitch](https://arxiv.org/abs/2006.06873))
- [Conformer](https://arxiv.org/abs/2005.08100)-based FastSpeech / FastSpeech2

You can find example configs of the above models in [`egs2/ljspeech/tts1/conf/tuning`](../../ljspeech/tts1/conf/tuning).

### Multi speaker model

- [X-Vector](https://ieeexplore.ieee.org/abstract/document/8461375) + Tacotron2
- X-Vector + Transformer-TTS
- X-Vector + FastSpeech
- X-Vector + FastSpeech2
- X-Vector + Conformer-based FastSpeech / FastSpeech2
- [GST](https://arxiv.org/abs/1803.09017) + Tacotron2
- GST + Transformer-TTS
- GST + FastSpeech
- GST + FastSpeech2
- GST + Conformer-based FastSpeech / FastSpeech2

X-Vector is provided by kaldi and pretrained with VoxCeleb corpus.
GST and X-vector can be combined (Not tested well).
You can find example configs of the above models in [`egs2/vctk/tts1/conf/tuning`](../../vctk/tts1/conf/tuning).

## FAQ

### ESPnet1 model is compatible with ESPnet2?

No. We cannot use the ESPnet1 model in ESPnet2.

### How to change minibatch size in training?

See [change mini-batch type](https://espnet.github.io/espnet/espnet2_training_option.html#change-mini-batch-type).
As a default, we use `batch_type=numel` and `batch_bins` instead of `batch_size` to enable us to use dynamic batch size.
See the following config as an example.
https://github.com/espnet/espnet/blob/96b2fd08d4fd9276aabd7ad41ec5e02a88b30958/egs2/ljspeech/tts1/conf/tuning/train_tacotron2.yaml#L61-L62

### How to make a new recipe for my own dataset?

See [how to make/port new recipe](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#how-to-makeport-new-recipe).

### How to add a new `g2p` module?

Update [`espnet2/text/phoneme_tokenizer.py`](https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py) to add new module.
Then, add new choice in the argument parser of [`espnet2/bin/tokenize_text.py`](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/espnet2/bin/tokenize_text.py#L226-L240) and [`espnet2/tasks/tts.py`](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/espnet2/tasks/tts.py#L180-L194).

We have the warpper module of [bootphon/phonemizer](https://github.com/bootphon/phonemizer).
You can find the module [`espnet2/text/phoneme_tokenizer.py`](https://github.com/kan-bayashi/espnet/blob/7cc12c6a25924892b281c2c1513de52365a1be0b/espnet2/text/phoneme_tokenizer.py#L110).
If the g2p you wanted is implemented in [bootphon/phonemizer](https://github.com/bootphon/phonemizer), we can easily add it [like this](https://github.com/kan-bayashi/espnet/blob/7cc12c6a25924892b281c2c1513de52365a1be0b/espnet2/text/phoneme_tokenizer.py#L172-L173) (Note that you need to update the choice as I mentioned the above).

### How to add a new `cleaner` module?

Update [`espnet2/text/cleaner.py`](https://github.com/espnet/espnet/blob/master/espnet2/text/cleaner.py) to add new module.
Then, add new choice in the argument parser of [`espnet2/bin/tokenize_text.py`](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/espnet2/bin/tokenize_text.py#L219-L225) and [`espnet2/tasks/tts.py`](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/espnet2/tasks/tts.py#L173-L179).

### How to use trained model in python?

See [use a pretrained model for inference](https://github.com/espnet/espnet_model_zoo#use-a-pretrained-model-for-inference).

### How to get pretrained models?

Use [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo).
You can find the all of the pretrained model list from [here](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv).

### How to load the pretrained model?

Please use `--init_param` option or add it in training config (`*.yaml`).

```bash
# Usage
--init_param <file_path>:<src_key>:<dst_key>:<exclude_keys>

# Load all parameters
python -m espnet2.bin.tts_train --init_param model.pth

# Load only the parameters starting with "decoder"
python -m espnet2.bin.tts_train --init_param model.pth:tts.dec

# Load only the parameters starting with "decoder" and set it to model.tts.dec
python -m espnet2.bin.tts_train --init_param model.pth:decoder:tts.dec

# Set parameters to model.tts.dec
python -m espnet2.bin.tts_train --init_param decoder.pth::tts.dec

# Load all parameters excluding "tts.enc.embed"
python -m espnet2.bin.tts_train --init_param model.pth:::tts.enc.embed

# Load all parameters excluding "tts.enc.embed" and "tts.dec"
python -m espnet2.bin.tts_train --init_param model.pth:::tts.enc.embed,tts.dec
```

### How to finetune the pretrained model?

See [jvs recipe example](../../jvs/tts1/README.md).

### How to add a new model?

Under construction.

### How to test my model with an arbitrary given text?

See Google Colab demo notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_tts_realtime_demo.ipynb)

### How to handle the errors in `validate_data_dir.sh`?

> `utils/validate_data_dir.sh: text contains N lines with non-printable characters which occurs at this line`

This is caused by the recent change in kaldi.
We recommend modifying the following part in `utils/validate_data_dir.sh` to be `non_print=true`.

https://github.com/kaldi-asr/kaldi/blob/40c71c5ee3ee5dffa1ad2c53b1b089e16d967bb5/egs/wsj/s5/utils/validate_data_dir.sh#L9

> `utils/validate_text.pl: The line for utterance xxx contains disallowed Unicode whitespaces`  
> `utils/validate_text.pl: ERROR: text file 'data/xxx' contains disallowed UTF-8 whitespace character(s)`

The use of zenkaku whitespace in `text` is not allowed.
Please changes it to hankaku whitespace or the other symbol.

### Why the model generate meaningless speech at the end?

This is because the model failed to predict the stop token.
There are several solutions to solve this issue:

- Use attention constraint in the inference (`use_attention_constraint=True` in inference config, only for Tacotron 2).
- Train the model with a large `bce_pos_weight` (e.g., `bce_pos_weight=10.0`).
- Use non-autoregressive models (FastSpeech or FastSpeech2)

### Why the model cannot be trained well with my own dataset?

The most of the problems are caused by the bad cleaning of the dataset.
Please check the following items carefully:

- Remove the silence at the beginning and end of the speech.
- Separate speech if it contains a long silence at the middle of speech.
- Use phonemes instead of characters if G2P is available.
- Clean the text as possible as you can (abbreviation, number, etc...)
- Add the pose symbol in text if the speech contains the silence.
- If the dataset is small, please consider the use of adaptation with pretrained model.
- If the dataset is small, please consider the use of large reduction factor, which helps the attention learning.
- Check the attention plot during the training. Loss value is not so meaningfull in TTS.

### How is the duration for FastSpeech2 generated?

We use the teacher model attention weight to calculate the duration as the same as FastSpeech.
See more info in [FastSpeech paper](https://arxiv.org/abs/1905.09263).

### Why the output of Tacotron2 or Transformer is non-deterministic?

This is because we use prenet in the decoder, which always applies dropout.
See more info in [Tacotron2 paper](https://arxiv.org/abs/1712.05884).
