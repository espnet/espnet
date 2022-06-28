# Speech Enhancement Frontend Recipe

This is the common recipe for ESPnet2 speech enhancement frontend. Currently, ten speech enhancement/separation recipes are supported:
```
egs2/
├── aishell4
│   └── enh1
├── chime4
│   └── enh1
├── clarity21
│   └── enh1
├── conferencingspeech21
│   └── enh1
├── dns_icassp21
│   └── enh1
├── dns_ins20
│   └── enh1
├── dns_ins21
│   └── enh1
├── librimix
│   └── enh1
├── sms_wsj
│   └── enh1
├── vctk_noisy
│   └── enh1
├── vctk_noisyreverb
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
In `egs2/TEMPLATE/enh1/enh.sh`, 13 stages are included.

#### Stage 1: Data preparation
This stage is similar to stage 1 in [asr.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh).

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
In `EnhancementTask`, the speech enhancement or separation models follow the `encoder-separator-decoder` style, and several encoders, decoders and separators are implemented. Although it is currently defined as an independent task, the models from `EnhancementTask` can be easily called by other tasks or even jointly trained with other tasks (see `egs2/TEMPLATE/enh_asr1/`, `egs2/TEMPLATE/enh_st1/`).

> Now we support adding noise and reverberation on the fly by specifying `--use_preprocessor` and `--extra_wav_list` to use `EnhPreprocessor`. Check [PR #4321](https://github.com/espnet/espnet/pull/4321#issue-1216290237) for more details.
>
> We also support possible integration of other speech enhancement/separation toolkits (e.g. [Asteroid](https://github.com/asteroid-team/asteroid)), so that models trained with other speech enhancement/separation toolkits can be reused/evaluated on ESPnet for downstream tasks such as ASR.

Related arguments in `enh.sh` include:

  + --spk_num
  + --enh_args
  + --enh_config
  + --enh_exp
  + --ngpu
  + --num_nodes
  + --init_param
  + --use_dereverb_ref
  + --use_noise_ref
  + --use_preprocessor
  + --extra_wav_list

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
│   │   ├── null_decoder.py
│   │   └── stft_decoder.py
│   ├── encoder
│   │   ├── abs_encoder.py
│   │   ├── conv_encoder.py
│   │   ├── null_encoder.py
│   │   └── stft_encoder.py
│   ├── espnet_model.py
│   ├── layers
│   │   ├── beamformer.py
│   │   ├── complex_utils.py
│   │   ├── complexnn.py
│   │   ├── conv_utils.py
│   │   ├── dc_crn.py
│   │   ├── dnn_beamformer.py
│   │   ├── dnn_wpe.py
│   │   ├── dpmulcat.py
│   │   ├── dprnn.py
│   │   ├── dptnet.py
│   │   ├── fasnet.py
│   │   ├── ifasnet.py
│   │   ├── mask_estimator.py
│   │   ├── skim.py
│   │   ├── tcn.py
│   │   └── wpe.py
│   ├── loss
│   │   ├── criterions
│   │   │   ├── abs_loss.py
│   │   │   ├── tf_domain.py
│   │   │   └── time_domain.py
│   │   └── wrappers
│   │       ├── abs_wrapper.py
│   │       ├── dpcl_solver.py
│   │       ├── fixed_order.py
│   │       ├── multilayer_pit_solver.py
│   │       └── pit_solver.py
│   └── separator
│       ├── abs_separator.py
│       ├── asteroid_models.py
│       ├── conformer_separator.py
│       ├── dan_separator.py
│       ├── dc_crn_separator.py
│       ├── dccrn_separator.py
│       ├── dpcl_separator.py
│       ├── dpcl_e2e_separator.py
│       ├── dprnn_separator.py
│       ├── dptnet_separator.py
│       ├── fasnet_separator.py
│       ├── neural_beamformer.py
│       ├── rnn_separator.py
│       ├── skim_separator.py
│       ├── svoice_separator.py
│       ├── tcn_separator.py
│       └── transformer_separator.py
└── tasks
    └── enh.py
```

#### Stage 7: Speech Enhancement inferencing
This stage generates the enhanced or separated speech with the trained model. The generated audio files will be placed at `${expdir}/${eval_set}/logdir` and scp files for them will be created in `${expdir}/${eval_set}`.

Related arguments in `enh.sh` include:

  + --spk_num
  + --fs
  + --gpu_inference
  + --inference_args
  + --inference_model
  + --inference_nj
  + --inference_enh_config

> Now we support changing the model attributes (such as `beamformer_type` in `NeuralBeamformer`) in this stage, by specifying `--inference_enh_config`. Check [PR #4251](https://github.com/espnet/espnet/pull/4251#issue-1199079132) for more details.

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

Related new python files:

```
egs2
└── TEMPLATE
    └── enh1
        └── scripts
            └── utils
                └── show_enh_score.sh
espnet2/
└── bin
    └── enh_scoring.py
```

#### Stage 9: Decode with a pretrained ASR model

Same as Stage 11 in the ASR task. The enhanced speech in Stage 7 is fed into a pretrained ASR model (specified as `"${asr_exp}"/"${decode_asr_model}"`) for decoding.

Related arguments in `enh.sh` include:

  + --score_with_asr
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

## (For developers) Instructions on creating a new recipe
#### Step 1 Create recipe directory
First, run the following command to create the directory for the new recipe from our template:
```bash
egs2/TEMPLATE/enh1/setup.sh egs2/<your_recipe_name>/enh1
```

> Please follow the name convention in other recipes.
>
> For the following steps, we assume the operations are done under the directory `egs2/<your_recipe_name>/enh1/`.

#### Step 2 Write scripts for data preparation
Prepare `local/data.sh`, which will be used in stage 1 in `enh.sh`.
It can take some arguments as input, see [egs2/wsj0_2mix/enh1/local/data.sh](https://github.com/espnet/espnet/blob/master/egs2/wsj0_2mix/enh1/local/data.sh) for reference.

The script `local/data.sh` should finally generate Kaldi-style data directories under `<recipe_dir>/data/`.
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

> Please follow the style in other recipes as much as possible. Check [egs2/chime4/enh1/local/data.sh](https://github.com/espnet/espnet/blob/master/egs2/chime4/enh1/local/data.sh) for reference.
>
> Remember to check your new scripts with `shellcheck`, otherwise they may fail the tests in [ci/test_shell.sh](https://github.com/espnet/espnet/blob/master/ci/test_shell.sh).

#### Step 3 Prepare training configuration
Prepare training configuration files (e.g. [train.yaml](https://github.com/espnet/espnet/blob/master/egs2/wsj0_2mix/enh1/conf/tuning/train_enh_rnn_tf.yaml)) under `conf/`.

> If you have multiple configuration files, it is recommended to put them under `conf/tuning/`, and create a symbolic link `conf/tuning/train.yaml` pointing to the config file with the best performance.
>
> Please trim trailing whitespace in each line.

#### Step 4 Prepare run.sh
Write `run.sh` to provide a template entry script, so that users can easily run your recipe by `./run.sh`.
Check [egs2/wsj0_2mix/enh1/run.sh](https://github.com/espnet/espnet/blob/master/egs2/wsj0_2mix/enh1/run.sh) for reference.

> Please ensure that the argument `--spk_num` in `run.sh` is consistent with the `num_spk` (under `separator_conf`) in the training configuration files created in last step.
>
> If your recipes provide references for noise and/or dereverberation, you can set the argument `--use_noise_ref true` and/or `--use_dereverb_ref true` in `run.sh`.

## Instructions on creating a new model
The current ESPnet-SE tool adopts an encoder-separator-decoder architecture for all models, e.g.

> For Time-Frequency masking models, the encoder and decoder would be [stft_encoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/encoder/stft_encoder.py) and [stft_decoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/decoder/stft_decoder.py) respectively, and the separator can be [dprnn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/dprnn_separator.py), [rnn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/rnn_separator.py), [tcn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tcn_separator.py), [transformer_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/transformer_separator.py) and so on.
> For TasNet, the encoder and decoder are [conv_encoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/encoder/conv_encoder.py) and [conv_decoder.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/decoder/conv_decoder.py) respectively. The separator is [tcn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tcn_separator.py).

#### Step 1 Create model scripts
For encoder, separator, and decoder models, create new scripts under [espnet2/enh/encoder/](https://github.com/espnet/espnet/tree/master/espnet2/enh/encoder), [espnet2/enh/separator/](https://github.com/espnet/espnet/tree/master/espnet2/enh/separator), and [espnet2/enh/decoder/](https://github.com/espnet/espnet/tree/master/espnet2/enh/decoder), respectively.

For a separator model, please make sure it implements the `num_spk` property. Check [espnet2/enh/separator/tcn_separator.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tcn_separator.py) for reference.

> Please follow the coding style as mentioned in [CONTRIBUTING.md](https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#41-python).
>
> Remember to format your new scripts to match the styles in `black` and `flake8`, otherwise they may fail the tests in [ci/test_python.sh](https://github.com/espnet/espnet/blob/master/ci/test_python.sh).

#### Step 2 Add the new model to related scripts
In [espnet2/tasks/enh.py](https://github.com/espnet/espnet/blob/master/espnet2/tasks/enh.py#L37-L62), add your new model to the corresponding `ClassChoices`, e.g.
* For encoders, add `<key>=<your-model>` to `encoder_choices`.
* For decoders, add `<key>=<your-model>` to `decoder_choices`.
* For separators, add `<key>=<your-model>` to `separator_choices`.

#### Step 3 [Optional] Create new loss functions
If you want to use a new loss function for your model, you can add it as a module to [espnet2/enh/loss/criterions/](https://github.com/espnet/espnet/blob/master/espnet2/enh/loss/criterions).
> Check `FrequencyDomainMSE` in [espnet2/enh/loss/criterions/tf_domain.py](https://github.com/espnet/espnet/blob/master/espnet2/enh/loss/criterions/tf_domain.py#L119) for reference.

Then add your loss name to `criterion_choices` in [espnet2/tasks/enh.py](https://github.com/espnet/espnet/blob/master/espnet2/tasks/enh.py#L103), so that you can configure it directly in a yaml file.

#### Step 4 Create unit tests for the new model
Finally, it would be nice to make some unit tests for your new model under [`test/espnet2/enh/test_espnet_model.py`](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/test_espnet_model.py) and [`test/espnet2/enh/encoder`](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/encoder) / [`test/espnet2/enh/decoder`](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/decoder) / [`test/espnet2/enh/separator`](https://github.com/espnet/espnet/tree/master/test/espnet2/enh/separator).
