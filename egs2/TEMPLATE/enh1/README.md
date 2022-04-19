# Speech Enhancement Frontend Recipe

This is the common recipe for ESPnet2 speech enhancement frontend. Currently, ten speech enhancement/separation recipes are supported:
```
egs2/
├── chime4
│   └── enh1
├── dirha_wsj
│   └── enh1
├── libri_css
│   └── enh1
├── librimix
│   └── enh1
├── reverb
│   └── enh1
├── sms_wsj
│   └── enh1
├── wham
│   └── enh1
├── whamr
│   └── enh1
├── wsj0_2mix
│   └── enh1
└── wsj0_2mix_spatialized
    └── enh1
```
## Introduction to enh.sh
In `egs2/TEMPLATE/enh1/enh.sh`, 12 stages are included.

#### Stage 1: Data preparation
This stage is the same as stage 1 in asr.sh.

#### Stage 2: Speech perturbation
Speech perturbation is widely used in ASR task, but rarely in speech enhancement. Some of our initial experiments have shown that speech perturbation works on `wsj0_2mix`. We are conducting more experiments to make sure if it works.
The speech perturbation procedure is almost the same as that in ASR, we have copied `scripts/utils/perturb_data_dir_speed.sh` to `scripts/utils/perturb_enh_data_dir_speed.sh` and made some minor modifications to support the speech perturbation for more scp files rather than `wav.scp` only.

#### Stage 3: Format wav.scp
Format scp files such as `wav.scp`. The scp files include:
  + `wav.scp`: wav file list of mixed/noisy input signals.
  + `spk{}.scp`: wav file list of speech reference signals. {} can be 1, 2, ..., depending on the number of speakers in the input signal in `wav.scp`.
  + `noise{}.scp` (optional): wav file list of noise reference signals. {} can be 1, 2, ..., depending on the number of noise types in the input signal in `wav.scp`. The file(s) are required when `--use_noise_ref true` is specified. Also related to the variable `noise_type_num`.
  + `dereverb{}.scp` (optional): wav file list of dereverberation reference signals (for training a dereverberation model). This file is required when `--use_dereverb_ref true` is specified. Also related to the variable `dereverb_ref_num`.
  + `utt2category`: (optional) the category info of each utterance. This file can help the batch sampler to load the same category utterances in each batch. One usage case is that users want to load the simulation data and real data in different batches.

#### Stage 4: Remove short data
This stage is same as that in ASR recipe.

#### Stage 5: Collect stats for enhancement task.
Same as the ASR task, we collect the data stats before training. Related new python files are:
```
espnet2/
├── bin
│   └── enh_train.py
└── tasks
    └── enh.py
```
The`EnhancementTask` defined in `espnet2/tasks/enh.py` is called in `espnet2/bin/enh_train.py`. In the `collect_stats` mode. the behavior of `EnhancementTask` is the same as the `ABSTask`.

#### Stage 6: Enhancement task Training
We have created `EnhancementTask` in `espnet2/tasks/enh.py`, which is used to train the `ESPnetEnhancementModel(AbsESPnetModel)` defined in `espnet2/enh/espnet_model.py`. 
In `EnhancementTask`, the speech enhancement or separation models follow the `encoder-separator-decoder` style, and several encoders, decoders and separators are implemented, Although it is currently defined as an independent task, the models from `EnhancementTask` can be easily called by ASR tasks or even jointly trained with ASR (`egs2/TEMPLATE/enh_asr1/`, will be merged in near future).

We are also working on possible integration of other speech enhancement/separation toolkits (e.g. [Asteroid](https://github.com/asteroid-team/asteroid)), so that models trained with other speech enhancement/separation toolkits can be reused/evaluated on ESPnet for downstream tasks such as ASR.

Related arguments in `enh.sh` include:

  + --enh_args
  + --enh_config
  + --enh_exp
  + --ngpu
  + --num_nodes
  + --init_param
  + --use_dereverb_ref
  + --use_noise_ref

Related python files:
```
espnet2/
├── bin
│   ├── enh_inference.py
│   ├── enh_scoring.py
│   └── enh_train.py
├── enh
│   ├── abs_enh.py
│   ├── decoder
│   │   ├── abs_decoder.py
│   │   ├── conv_decoder.py
│   │   └── stft_decoder.py
│   ├── encoder
│   │   ├── abs_encoder.py
│   │   ├── conv_encoder.py
│   │   └── stft_encoder.py
│   ├── espnet_model.py
│   ├── layers
│   │   ├── beamformer.py
│   │   ├── dnn_beamformer.py
│   │   ├── dnn_wpe.py
│   │   ├── dprnn.py
│   │   ├── mask_estimator.py
│   │   └── tcn.py
│   └── separator
│       ├── abs_separator.py
│       ├── dprnn_separator.py
│       ├── neural_beamformer.py
│       ├── rnn_separator.py
│       ├── tcn_separator.py
│       └── transformer_separator.py
└── tasks
        └── enh.py
```

#### Stage 7: Speech Enhancement inferencing
This stage generates the enhanced or separated speech with the trained model. The generated audio files will be placed at `${expdir}/${eval_set}/logdir` and scp files for them will be created in `${expdir}/${eval_set}`.

Related arguments in `enh.sh` include:

  + --gpu_inference
  + --inference_args
  + --inference_model
  + --inference_nj

Related new python files:
```
espnet2/
└── bin
    └── enh_inference.py
```

#### Stage 8: Scoring

This stage is used to do the scoring for speech enhancement. Scoring results for each `${eval_set}` will be summarized in `${expdir}/RESULTS.TXT`.

Related arguments in `enh.sh` include:

  + --scoring_protocol
  + --ref_channel
  + --score_with_asr

Related new python files:

```
espnet2/
└── bin
    └── enh_scoring.py
```

#### Stage 9: Decode with a pretrained ASR model

Same as Stage 11 in the ASR task. The enhanced speech in Stage 7 is fed into a pretrained ASR model (specified as `"${asr_exp}"/"${decode_asr_model}"`) for decoding.

Related arguments in `enh.sh` include:

  + --asr_exp
  + --decode_args
  + --decode_asr_model
  + --gpu_inference
  + --inference_nj

Related python files:

```
espnet2/
└── bin
    └── asr_inference.py
```

#### Stage 10: Scoring with a pretrained ASR model

Same as Stage 12 in the ASR task. The decoding results in Stage 11 are scored to calculate the average CER/WER/TER.

#### Stage 11: Pack model

Just the same as other tasks. A new entry for packing speech enhancement models is added in `espnet2/bin/pack.py`.

#### Stage 12: Upload model to Zenodo (Deprecated)

Upload the trained speech enhancement/separation model to Zenodo for sharing.

#### Stage 13: Upload model to Hugging Face

Upload the trained speech enhancement/separation model to Hugging Face for sharing. Additonal information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## Instructions on creating a new recipe
#### Step 1 Create recipe directory
First, run the following command to create the directory for the new recipe from our template:
```bash
egs2/TEMPLATE/enh1/setup.sh egs2/<your-recipe-name>/enh1
```

> For the following steps, we assume the operations are done under the directory `egs2/<your-recipe-name>/enh1/`.

#### Step 2 Write scripts for data preparation
Prepare `local/data.sh`, which will be used in stage 1 in `enh.sh`.
It can take some arguments as input, see [egs2/wsj0_2mix/enh1/local/data.sh](https://github.com/espnet/espnet/blob/master/egs2/wsj0_2mix/enh1/local/data.sh) for reference.

The script `local/data.sh` should finally generate Kaldi-style data directories under `<recipe-dir>/data/`.
Each subset directory should contains at least 4 files:
```
<recipe-dir>/data/<subset-name>/
├── spk1.scp   (clean speech references)
├── spk2utt
├── utt2spk
└── wav.scp    (noisy speech)
```
Optionally, it can also contain `noise{}.scp` and `dereverb{}.scp`, which point to the corresponding noise and dereverberated references respectively. {} can be 1, 2, ..., depending on the number of noise types (dereverberated signals) in the input signal in `wav.scp`.

Make sure to sort the scp and other related files as in Kaldi. Also, remember to run `. ./path.sh` in `local/data.sh` before sorting, because it will force sorting to be byte-wise, i.e. `export LC_ALL=C`.

> Remember to check your new scripts with `shellcheck`, otherwise they may fail the tests in [ci/test_shell.sh](https://github.com/espnet/espnet/blob/master/ci/test_shell.sh).

#### Step 3 Prepare training configuration
Prepare training configuration files (e.g. [train.yaml](https://github.com/espnet/espnet/blob/master/egs2/wsj0_2mix/enh1/conf/tuning/train_enh_rnn_tf.yaml)) under `conf/`.

> If you have multiple configuration files, it is recommended to put them under `conf/tuning/`, and create a symbolic link `conf/tuning/train.yaml` pointing to the config file with the best performance.

#### Step 4 Prepare run.sh
Write `run.sh` to provide a template entry script, so that users can easily run your recipe by `./run.sh`.
See [egs2/wsj0_2mix/enh1/run.sh](https://github.com/espnet/espnet/blob/master/egs2/wsj0_2mix/enh1/run.sh) for reference.

> If your recipes provide references for noise and/or dereverberation, you can add the argument `--use_noise_ref true` and/or `--use_dereverb_ref true` in `run.sh`.

## Instructions on creating a new model
The current ESPnet-SE tool adopts an encoder-separator-decoder architecture for all models, e.g.

> For Time-Frequency masking models, the encoder and decoder would be [stft_encoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/encoder/stft_encoder.py) and [stft_decoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/decoder/stft_decoder.py) respectively, and the separator can be any of [dprnn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/dprnn_separator.py), [rnn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/rnn_separator.py), [tcn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tcn_separator.py), and [transformer_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/transformer_separator.py).
> For TasNet, the encoder and decoder are [conv_encoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/encoder/conv_encoder.py) and [conv_decoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/decoder/conv_decoder.py) respectively. The separator is [tcn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tcn_separator.py).

#### Step 1 Create model scripts
For encoder, separator, and decoder models, create new scripts under [espnet2/enh/encoder/](https://github.com/espnet/espnet/tree/master/espnet2/enh/encoder), [espnet2/enh/separator/](https://github.com/espnet/espnet/tree/master/espnet2/enh/separator), and [espnet2/enh/decoder/](https://github.com/espnet/espnet/tree/master/espnet2/enh/decoder), respectively.

For a separator model, please make sure it implements the `num_spk` property. See [espnet2/enh/separator/rnn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/rnn_separator.py) for reference.

> Remember to format your new scripts to match the styles in `black` and `flake8`, otherwise they may fail the tests in [ci/test_python.sh](https://github.com/espnet/espnet/blob/master/ci/test_python.sh).

#### Step 2 Add the new model to related scripts
In [espnet2/tasks/enh.py](https://github.com/espnet/espnet/blob/master/espnet2/tasks/enh.py#L37-L62), add your new model to the corresponding `ClassChoices`, e.g.
* For encoders, add `<key>=<your-model>` to `encoder_choices`.
* For decoders, add `<key>=<your-model>` to `decoder_choices`.
* For separators, add `<key>=<your-model>` to `separator_choices`.

#### Step 3 [Optional] Create new loss functions
If you want to use a new loss function for your model, you can add it to [espnet2/enh/espnet_model.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/espnet_model.py), such as:
```python
    @staticmethod
    def new_loss(ref, inf):
        """Your new loss
        Args:
            ref: (Batch, samples)
            inf: (Batch, samples)
        Returns:
            loss: (Batch,)
        """
        ...
        return loss
```

Then add your loss name to [ALL_LOSS_TYPES](https://github.com/espnet/espnet/blob/master/espnet2/enh/espnet_model.py#L21), and handle the loss calculation in [_compute_loss](https://github.com/espnet/espnet/blob/master/espnet2/enh/espnet_model.py#L246).

#### Step 4 Create unit tests for the new model
Finally, it would be nice to make some unit tests for your new model under [test/espnet2/enh/encoder](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/encoder), [test/espnet2/enh/decoder](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/decoder), or [test/espnet2/enh/separator](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/separator).
