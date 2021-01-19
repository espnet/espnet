# Speech Enhancement Frontend Recipe

This is the common recipe for for ESPnet2 speech enhancement frontend. Currently, ten speech enhancement/separation recipes are supported:
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
In `egs2/TEMPLATE/enh1/enh.sh`, 12 stages are included.

#### Stage 1: Data preparation
This stage is same as stage 1 in asr.sh.

#### Stage 2: Speech perturbation
Speech perturbation is widely used in ASR task, but rarely in speech enhancement. Some of our initial experiments have shown that speech perturbation works on `wsj0_2mix`. We are conducting more experiment to make sure if it works.
The speech perturbation procedure is almost same as that in ASR, we have copied `scripts/utils/perturb_data_dir_speed.sh` to `scripts/utils/perturb_enh_data_dir_speed.sh` and made minor modification to support the speech perturbation for more scp files rather than `wav.scp` only.

#### Stage 3: Format wav.scp
Format scp files such as `wav.scp`. The scp files include:
  + `wav.scp`: wav file list of mixed/noisy input signals.
  + `spk{}.scp`: wav file list of speech reference signals. {} can be 1, 2, ..., depending on the number of speakers in the input signal in `wav.scp`.
  + `noise{}.scp` (optional): wav file list of noise reference signals. {} can be 1, 2, ..., depending on the number of noise types in the input signal in `wav.scp`. The file(s) are required when `--use_noise_ref true` is specified. Also related to the variable `noise_type_num`.
  + `dereverb{}.scp` (optional): wav file list of dereverberation reference signals (for training a dereverberation model). This file is required when `--use_dereverb_ref true` is specified. Also related to the variable `dereverb_ref_num`.

#### Stage 4: Remove short data
This stage is same as that in ASR recipe.

#### Stage 5: Collect stats for enhancement task.
Same as the ASR task, we collet the data stats before training. Related new python files are:
```
espnet2/
├── bin
│   └── enh_train.py
└── tasks
    └── enh.py
```
The`EnhancementTask` defined in `espnet2/tasks/enh.py` is called in `espnet2/bin/enh_train.py`. In the `collect_stats` mode. the behavior of `EnhancementTask` is same as the `ABSTask`.

#### Stage 6: Enhancemnt task Training
We have created `EnhancementTask` in `espnet2/tasks/enh.py`, which is used to train the `ESPnetEnhancementModel(AbsESPnetModel)` defined in `espnet2/enh/espnet_model.py`. 
In `EnhancementTask`, the speech enhancement or separation models follow the `encoder-separator-decoder` style, and several encoders, decoders and separators are implemented, Although it is currently defined as an independent task, the models from `EnhancementTask` can be easily called by ASR tasks or even jointly trained with ASR (`egs2/TEMPLATE/enh_asr1/`, will be merged in near future).

We are also working on possible interagtion of other speech enhancement/separation toolkits (e.g. [Asteroid](https://github.com/asteroid-team/asteroid)), so that models trained with other speech enhancement/separation toolkits can be reused/evaluated on ESPnet for downstream tasks such as ASR.

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

Just same as other tasks. New entry for packing speech enhancement models is added in `espnet2/bin/pack.py`.

#### Stage 12: Upload model to Zenodo

Upload the trained speech enhancement/separation model to Zenodo for sharing.
