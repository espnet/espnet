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
    * [How to fine-tune pre-trained OWSM](#how-to-fine-tune-pre-trained-owsm)
  * [Related work](#related-work)

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


### How to fine-tune pre-trained OWSM

Pre-trained OWSM can be fine-tuned on a specific dataset. Here, we use AISHELL-1 as an example.

#### 1. Prepare `s2t1` recipe

We use this `s2t1` template to fine-tune OWSM. So we first create this directory under our custom dataset `egs2/aishell`.

```bash
egs2/TEMPLATE/s2t1/setup.sh egs2/aishell/s2t1
```

Then, we download a pre-trained model, e.g., `espnet/owsm_v2_ebranchformer`, using the following command:

```bash
# go to the created dir
cd egs2/aishell/s2t1

# source path.sh
. ./path.sh

# download model from hugging face using espnet_model_zoo_download
# we use dummy names for required arguments and we do not run any actual stage (thus --stage 100)
./s2t.sh --download_model espnet/owsm_v2_ebranchformer --stage 100 --train_set dummy --valid_set dummy2 --test_sets dummy3
```

The downloaded model will be saved in local cache and then uncompressed. An `exp` directory will be automatically created which contains symbolic links to the checkpoint and config files.

To use a pre-trained model, we need the following important files:
- config: A yaml file containing all training arguments and the token list. The name is `config.yaml`.
- model checkpoint: The name is `xxx.pth`. In this example, it is `valid.total_count.ave_5best.till25epoch.pth`.
- stats: This is used to normalize the input speech features if `feats_normalize` is `global_mvn`.
- bpe model: This is the BPE model used by `sentencepiece`.

The path to `stats` can be found in `config.yaml`, e.g.:
```bash
grep stats_file exp/espnet/owsm_v2_ebranchformer/config.yaml
```

The path to `bpemodel` can also be found in `config.yaml`, e.g.:
```bash
grep bpemodel exp/espnet/owsm_v2_ebranchformer/config.yaml
```

In the following sections, we will manually copy those two files to correct places.

#### 2. Prepare data in OWSM format

The data should be prepared in the OWSM format. Please refer to [1\. Data preparation](#1-data-preparation) for more information.

Since AISHELL-1 has been included in OWSM v1, we can reuse those preparation scripts. For your own data, please write the scripts by yourself and make sure the special tokens such as the language codes are consistent with the pre-trained model. Note that we will NOT generate new vocabulary for fine-tuning. Instead, we will use the vocabulary from the pre-trained model.

```bash
cd local/
ln -s ../../../mixed_v1/s2t1/local/utils.py ./
ln -s ../../../mixed_v1/s2t1/local/prepare_aishell.* ./
cd ..

# modify data_dir and execute:
./local/prepare_aishell.sh
```

The prepared data will be stored in a new directory `data`.

Next, we execute various stages in `s2t.sh`. To make it easier, we create a `run.sh` shown below. It is mostly copied from the OWSM v2 recipe.

```bash
#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=AISHELL-1/train
valid_set=AISHELL-1/dev
test_sets="AISHELL-1/dev"

nbpe=50000  # this should be consistent with the pre-trained model
s2t_config=conf/train_s2t_ebf_conv2d_size1024_e12_d12.yaml
inference_config=conf/decode_s2t.yaml

# inference only args
# --cleaner whisper_basic --hyp_cleaner whisper_basic
./s2t.sh \
    --stage 3 \
    --stop_stage 4 \
    --use_lm false \
    --num_nodes 1 \
    --ngpu 4 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
    --num_splits_s2t 1 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 10000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"

```

We run Stage 3 and Stage 4 to format data:
```bash
./run.sh --stage 3 --stop_stage 4
```

We create the BPE token directory by ourselves. This is equivalent to Stage 5 but we do not generate a new token list.
```bash
mkdir -p data/token_list/bpe_unigram50000
cp path_to_bpe_model data/token_list/bpe_unigram50000 # path_to_bpe_model is in config.yaml

# we extract the token list
python -c "import yaml; config = yaml.safe_load(open('exp/espnet/owsm_v2_ebranchformer/config.yaml', 'r')); open('data/token_list/bpe_unigram50000/tokens.txt', 'w').write('\n'.join(config['token_list'])
)"
```

#### 3. Fine-tune the model

We create a training config file for fine-tuning. It is modified from the original config in `config.yaml`.
Note that you may need to tune the training hyperparameters such as learning rate. The model might easily overfit to a small training set.

```yaml
preprocessor: s2t
preprocessor_conf:
    text_prev_name: text_prev
    text_ctc_name: text_ctc
    fs: 16000
    na_symbol: "<na>"
    speech_length: 30
    speech_resolution: 0.02
    speech_init_silence: 30
    text_prev_apply_prob: 0.0   # we do not use previous prompt
    time_apply_prob: 0.0    # we do not use any timestamp for fine-tuning
    notime_symbol: "<notimestamps>"
    first_time_symbol: "<0.00>"
    last_time_symbol: "<30.00>"

frontend_conf:
    n_fft: 512
    win_length: 400
    hop_length: 160

specaug: specaug
specaug_conf:
    apply_time_warp: false
    time_warp_window: 5
    time_warp_mode: bicubic
    apply_freq_mask: true
    freq_mask_width_range:
    - 0
    - 27
    num_freq_mask: 2
    apply_time_mask: true
    time_mask_width_ratio_range:
    - 0.
    - 0.05
    num_time_mask: 10

encoder: e_branchformer
encoder_conf:
    output_size: 1024
    attention_heads: 16
    attention_layer_type: selfattn
    pos_enc_layer_type: abs_pos
    rel_pos_type: latest
    cgmlp_linear_units: 4096
    cgmlp_conv_kernel: 31
    use_linear_after_conv: false
    gate_activation: identity
    num_blocks: 12
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.1
    input_layer: conv2d
    layer_drop_rate: 0.0
    linear_units: 4096
    positionwise_layer_type: linear
    use_ffn: true
    macaron_ffn: true
    merge_conv_kernel: 31

decoder: transformer
decoder_conf:
    attention_heads: 16
    linear_units: 4096
    num_blocks: 12
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.1
    src_attention_dropout_rate: 0.1

model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1
    length_normalized_loss: false
    sym_na: "<na>"

# NOTE: you may need to tune these hyperparams
optim: adamw
optim_conf:
    lr: 1.0e-04
    betas:
    - 0.9
    - 0.98
    eps: 1.0e-06
    weight_decay: 0.0
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 5000

# NOTE: we are using 4 GPUs with 48GB memory
batch_type: unsorted
batch_size: 16
accum_grad: 4
max_epoch: 20
patience: none
init: none
best_model_criterion:
-   - valid
    - acc
    - max
-   - valid
    - total_count
    - max
keep_nbest_models: 5
use_amp: true
num_workers: 4
unused_parameters: false
seed: 2023
num_att_plot: 1

# fine-tune
init_param:
- exp/espnet/owsm_v2_ebranchformer/valid.total_count.ave_5best.till25epoch.pth
ignore_init_mismatch: false

```

We need to collect the shapes of speech and text, but skip the mean and variance collection because we use a pre-trained model. We run Stage 10 with a smaller batch size:
```bash
./run.sh --stage 10 --stop_stage 10 --feats_normalize utterance_mvn --s2t_args "--model_conf extract_feats_in_collect_stats=false --batch_size 5"
```

Then, we copy the existing mean and variance to the correct place:
```bash
cp path_to_train_stats exp/s2t_stats_raw_bpe50000/train/  # path_to_train_stats is in config.yaml
```

Now, we can start training:
```bash
./run.sh --stage 11 --stop_stage 11
```

## Related work
```
@article{peng2023reproducing,
  title={Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data},
  author={Peng, Yifan and Tian, Jinchuan and Yan, Brian and Berrebbi, Dan and Chang, Xuankai and Li, Xinjian and Shi, Jiatong and Arora, Siddhant and Chen, William and Sharma, Roshan and others},
  journal={arXiv preprint arXiv:2309.13876},
  year={2023}
}
```
