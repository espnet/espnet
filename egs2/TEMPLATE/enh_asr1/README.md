# Speech Enhancement Frontend Recipe

This is the common recipe for ESPnet2 joint-task with speech enhancement frontend. Following are the directory structure of speech enhancement and joint-task recipes:

```
egs2/
├── chime4/
│   ├── enh1/
│   ├── enh_asr1/
│   └── asr1/
├── l3das22/
│   └── enh1/
|   │   ├── conf/
|   │   ├── local/
|   |   │   ├── data.sh
|   |   │   ├── metric.sh
│   |   │   └── ...
|   │   ├── enh.sh -> ../../TEMPLATE/enh1/enh.sh
|   │   ├── run.sh
|   │   └── ...
├── lt_slurp_spatialized/
│   └── enh1/
├── slurp_spatialized/
│   ├── enh_asr1/
|   │   ├── enh_asr.sh -> ../../TEMPLATE/enh_asr1/enh_asr.sh
|   │   ├── run.sh
|   │   └── ...
│   └── asr1/
├── ...
└── TEMPLATE/
    ├── enh1/
    │   └── enh.sh
    ├── enh_asr1/
    │   └── enh_asr.sh
    ├── enh_diar1/
    │   └── enh_diar.sh
    ├── enh_st1/
    │   └── enh_st.sh
    └── ...
``` 


- stage 1 to stage 5: data preparation stages

- stage 6 to stage 9: language model training steps

- stage 10 to stage 11: joint-task training steps

- stage 12 to stage 13: Inference stages: Decoding and enhancing

- stage 14 to stage 15: Scoring recognition and SSE results 

- stage 16 to stage 17: model uploading steps


## Introduction to enh.sh
In `egs2/TEMPLATE/enh1/enh.sh`, 13 stages are included.

#### Stage 1: Data preparation
This stage is similar to stage 1 in [asr.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh).

#### Stage 2: Speech perturbation
Speech perturbation is widely used in the ASR task, but rarely in speech enhancement. Some of our initial experiments have shown that speech perturbation works on `wsj0_2mix`. We are conducting more experiments to make sure if it works.
The speech perturbation procedure is almost the same as that in ASR, we have copied `scripts/utils/perturb_data_dir_speed.sh` to `scripts/utils/perturb_enh_data_dir_speed.sh` and made some minor modifications to support the speech perturbation for more scp files rather than `wav.scp` only.

#### Stage 3: Format wav.scp
Format scp files such as `wav.scp`. The scp files include:
  + `wav.scp`: wav file list of mixed/noisy input signals.
  + `spk{}.scp`: wav file list of speech reference signals. {} can be 1, 2, ..., depending on the number of speakers in the input signal in `wav.scp`.
  + `noise{}.scp` (optional): wav file list of noise reference signals. {} can be 1, 2, ..., depending on the number of noise types in the input signal in `wav.scp`. The file(s) are required when `--use_noise_ref true` is specified. Also related to the variable `noise_type_num`.
  + `dereverb{}.scp` (optional): wav file list of dereverberation reference signals (for training a dereverberation model). This file is required when `--use_dereverb_ref true` is specified. Also related to the variable `dereverb_ref_num`.
  + `utt2category`: (optional) the category info of each utterance. This file can help the batch sampler to load the same category utterances in each batch. One usage case is that users want to load the simulation data and real data in different batches.

#### Stage 4: Remove short data
This stage is the same as that in ASR recipes.

#### Stage 5: Generate token_list using BPE

#### Stage 6: LM collect stats

#### Stage 7: LM Training

#### Stage 8: Calc perplexity

#### Stage 9: Ngram Training

#### Stage 10: Collect stats for the joint task.

Same as the ASR task, we collect the data stats before training. Related new python files are:
```
espnet2/
├── bin
│   └── enh_train.py
└── tasks
    └── enh.py
```
The`EnhancementTask` defined in `espnet2/tasks/enh.py` is called in `espnet2/bin/enh_train.py`. In the `collect_stats` mode. the behavior of `EnhancementTask` is the same as the `ABSTask`.


#### Stage 11: Joint task Training
We have created `EnhancementTask` in `espnet2/tasks/enh.py`, which is used to train the `ESPnetEnhancementModel(AbsESPnetModel)` defined in `espnet2/enh/espnet_model.py`.
In `EnhancementTask`, the speech enhancement or separation models follow the `encoder-separator-decoder` style, and several encoders, decoders and separators are implemented. Although it is currently defined as an independent task, the models from `EnhancementTask` can be easily called by other tasks or even jointly trained with other tasks (see `egs2/TEMPLATE/enh_asr1/`, `egs2/TEMPLATE/enh_st1/`).

Related python files:
```
espnet2/
├── bin/
│   ├── asr_inference.py
│   ├── diar_inference.py
│   ├── enh_s2t_train.py
│   ├── st_inference.py
│   └── ...
├── enh/
│   ├── espnet_enh_s2t_model.py
│   └── ...
├── tasks/
│   ├── enh_s2t.py
│   └── ...
└── ...
```


#### Stage 12: Decoding

#### Stage 13: Enhance Speech
This stage generates the enhanced or separated speech with the trained model. The generated audio files will be placed at `${expdir}/${eval_set}/logdir` and scp files for them will be created in `${expdir}/${eval_set}`.

Related arguments in `enh.sh` include:

  + --ref_num
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


#### Stage 14: Scoring ASR

#### Stage 15: Scoring Enhancement

This stage is used to do the scoring for speech enhancement. Scoring results for each `${eval_set}` will be summarized in `${expdir}/RESULTS.TXT`.

Related arguments in `enh.sh` include:

  + --scoring_protocol
  + --ref_channel

Related python files:

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


#### Stage 16: Pack model

Just the same as other tasks. A new entry for packing speech enhancement models is added in `espnet2/bin/pack.py`.

#### Stage 17: Upload model to Hugging Face

Upload the trained speech enhancement/separation model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).
