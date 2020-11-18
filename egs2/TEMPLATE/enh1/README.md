# Speech Enhancement Frontend Recipe

This is the common recipe for for ESPnet2 speech enhancement frontend. Currently, two speech separation recipes (`wsj0_2mix` and `wsj0_2mix_spatialized`) are supported:
```
egs2/
├── wsj0_2mix
│   └── enh1
└── wsj0_2mix_spatialized
    └── enh1
```
In `egs2/TEMPLATE/enh1/enh.sh`, 9 stages are included.

#### Stage 1: Data preparation
This stage is same as stage 1 in asr.sh.

#### Stage 2: Speech perturbation
Speech perturbation is widely used in ASR task, but rarely in speech enhancement. Some of our initial experiments have shown that speech perturbation works on `wsj0_2mix`. We are conducting more experiment to make sure if it works.
The speech perturbation procedure is almost same as that in ASR, I have copied `scripts/utils/perturb_data_dir_speed.sh` to `scripts/utils/perturb_enh_data_dir_speed.sh` and made minor modification to support the speech perturbation for more scp files rather than `wav.scp` only.

#### Stage 3: Format wav.scp
Format scp files such as `wav.scp`. The scp files include:
  + `wav.scp`: wav file list of mixed/noisy input signals.
  + `spk{}.scp`: wav file list of speech reference signals. {} can be 1, 2, ..., depending on the number of speakers in the input signal in `wav.scp`.
  + `noise{}.scp` (optional): wav file list of noise reference signals. {} can be 1, 2, ..., depending on the number of noise types in the input signal in `wav.scp`. The file(s) are required when `--use_noise_ref true` is specified.
  + `dereverb.scp` (optional): wav file list of dereverberation reference signals (for training a dereverberation model). This file is required when `--use_dereverb_ref true` is specified.

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
The`EnhancementTask` defined in `enh.py` is called in `enh_train.py`. In the `collect_stats` mode. the behavior of `EnhancementTask` is same as the `ABSTask`.

#### Stage 6: Enhancemnt task Training
We have created `EnhancementTask` in `espnet2/tasks/enh.py`, which is used to train the `ESPnetEnhancementModel(AbsESPnetModel)` defined in `espnet2/enh/espnet_model.py`. 
In `EnhancementTask`, several speech enhancement or separation models have been implemented (see `espnet2/enh/nets/`). Although it is currently defined as an independent task, the models from `EnhancementTask` can be easily called by ASR tasks or even jointly trained with ASR in the future (see espnets/asr/frontend/default.py).

Related python files:
```
espnet2/
├── bin
│   └── enh_train.py
├── enh
│   ├── __init__.py
│   ├── abs_enh.py
│   ├── espnet_model.py
│   ├── funcs
│   │   ├── __init__.py
│   │   ├── conv_beamformer.py
│   │   ├── dnn_beamformer.py
│   │   └── dnn_wpe.py
│   └── nets
│       ├── __init__.py
│       ├── beamformer_net.py
│       ├── tasnet.py
│       └── tf_mask_net.py
└── tasks
    └── frontend.py
```

#### Stage 7: Speech Enhancement inferencing
This stage generates the enhanced or separated speech with the trained model. The generated audio files will be placed at `${expdir}/${eval_set}/logdir` and scp files for them will be created in `${expdir}/${eval_set}`.

Related new python files:
```
espnet2/
└── bin
    └── enh_inference.py
```

#### Stage 8: Scoring

This stage is used to do the scoring for speech enhancement. Scoring results for each `${eval_set}` will be summarized in `${expdir}/RESULTS.TXT`

Related new python files:

```
espnet2/
└── bin
    └── enh_scoring.py
```

#### Stage 9: Pack model

Just same as other tasks. New entry for packing speech enhancement models is added in `espnet2/bin/pack.py`.